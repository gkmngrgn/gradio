"""Shared, owner-scoped file storage for multi-replica Gradio apps.

Gradio normally keeps uploaded and generated files in one process's temp
directory. Behind a load balancer without affinity, a file uploaded to replica A
is a 404 on replica B. This module is the seam that lets file bytes and their
owner live in shared storage, while the local temp directory stays the default.

The object store is a Hugging Face Storage Bucket (S3-like, via
``huggingface_hub``), which is already a core dependency. Bytes are addressed by
the path relative to the upload directory, so every replica derives the same key
from the same client-supplied path; the ownership record is committed with the
bytes and checked on every read, so a file is never served to another principal.

With no store configured the in-process behavior is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

FILE_STORE_ENV_VAR = "GRADIO_FILE_STORE"


@dataclass
class FileRecord:
    """A stored file's ownership and bookkeeping.

    ``key`` is the path relative to the upload directory, the same on every
    replica. ``local_path`` is set only by an in-process store, where the bytes
    never moved.
    """

    key: str
    owner: str | None = None
    session_hash: str | None = None
    committed_at: float = 0.0
    size: int = 0
    local_path: str | None = None


@runtime_checkable
class FileStore(Protocol):
    """The file-storage seam. One implementation per backend."""

    def put(
        self,
        local_path: str,
        key: str,
        *,
        owner: str | None,
        session_hash: str | None = None,
    ) -> FileRecord:
        """Commit bytes and their ownership before the upload is acknowledged."""

    def resolve(self, key: str, principal: str | None) -> FileRecord | None:
        """Return the record, or ``None`` on a miss or an ownership mismatch."""

    def materialize(self, key: str, principal: str | None) -> str | None:
        """A local path for the bytes if ``principal`` owns them, else ``None``."""

    def delete(self, key: str) -> None:
        """Remove a stored file."""

    def list_records(self) -> list[FileRecord]:
        """Every stored file with its ownership metadata, for orphan GC."""


def _record(
    key: str, local_path: str, owner: str | None, session_hash: str | None
) -> FileRecord:
    try:
        size = os.path.getsize(local_path)
    except OSError:
        size = 0
    return FileRecord(
        key=key,
        owner=owner,
        session_hash=session_hash,
        committed_at=time.time(),
        size=size,
        local_path=local_path,
    )


class LocalFileStore:
    """The in-process implementation: bytes stay where they are.

    Ownership is kept in memory, matching the default single-process semantics
    (no cross-principal isolation beyond the client session identity).
    """

    def __init__(self, root: str | None = None):
        self.root = root
        self._records: dict[str, FileRecord] = {}
        self._lock = threading.Lock()

    def put(self, local_path, key, *, owner, session_hash=None):
        record = _record(key, local_path, owner, session_hash)
        with self._lock:
            self._records[key] = record
        return record

    def resolve(self, key, principal):
        record = self._records.get(key)
        if record is None or record.owner != principal:
            return None
        return record

    def materialize(self, key, principal):
        record = self.resolve(key, principal)
        if record is None:
            return None
        return record.local_path if os.path.exists(record.local_path or "") else None

    def delete(self, key):
        with self._lock:
            self._records.pop(key, None)

    def list_records(self) -> list[FileRecord]:
        with self._lock:
            return list(self._records.values())


class HfBucketFileStore:
    """Owner-scoped files in a Hugging Face Storage Bucket.

    Bytes live at ``{prefix}/{app_id}/{key}`` and the ownership record at
    ``{prefix}/owners/{app_id}/{key}.json``. The client's path is unchanged:
    every replica derives the same relative key from it.
    """

    def __init__(
        self,
        bucket: str,
        token: str | None = None,
        *,
        app_id: str = "gradio",
        prefix: str = "files",
        client: Any = None,
    ):
        self.bucket = bucket
        self.token = token
        self.app_id = app_id
        self.prefix = prefix
        self._client = client

    def _api(self):
        if self._client is not None:
            return self._client
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def _object_key(self, key: str) -> str:
        return f"{self.prefix}/{self.app_id}/{key.lstrip('/')}"

    def _owner_key(self, key: str) -> str:
        return f"{self.prefix}/owners/{self.app_id}/{key.lstrip('/')}.json"

    def put(self, local_path, key, *, owner, session_hash=None):
        record = _record(key, local_path, owner, session_hash)
        payload = json.dumps(
            {
                "owner": record.owner,
                "session_hash": record.session_hash,
                "committed_at": record.committed_at,
                "size": record.size,
            }
        ).encode()
        self._api().batch_bucket_files(
            bucket_id=self.bucket,
            add=[
                (local_path, self._object_key(key)),
                (payload, self._owner_key(key)),
            ],
        )
        return record

    def _owner_record(self, key: str) -> dict[str, Any] | None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = str(Path(tmp) / "owner.json")
            self._api().download_bucket_files(
                bucket_id=self.bucket, files=[(self._owner_key(key), dest)]
            )
            if not os.path.exists(dest):
                return None
            with open(dest, encoding="utf-8") as fh:
                return json.load(fh)

    def resolve(self, key, principal):
        try:
            owner = self._owner_record(key)
        except Exception:  # a missing or unreadable record is a miss, not an error
            return None
        if owner is None or owner.get("owner") != principal:
            return None
        return FileRecord(
            key=key,
            owner=owner.get("owner"),
            session_hash=owner.get("session_hash"),
            committed_at=owner.get("committed_at", 0.0),
            size=owner.get("size", 0),
        )

    def materialize(self, key, principal):
        if self.resolve(key, principal) is None:
            return None
        # ponytail: download-to-temp, no streaming/range and no delete. Move to a
        # streaming response if large-file serving or disk pressure matters.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(key).suffix)
        tmp.close()
        self._api().download_bucket_files(
            bucket_id=self.bucket,
            files=[(self._object_key(key), tmp.name)],
        )
        return tmp.name

    def delete(self, key):
        self._api().batch_bucket_files(
            bucket_id=self.bucket,
            delete=[self._object_key(key), self._owner_key(key)],
        )

    def _owner_prefix(self) -> str:
        return f"{self.prefix}/owners/{self.app_id}/"

    def list_records(self) -> list[FileRecord]:
        prefix = self._owner_prefix()
        records = []
        for entry in self._api().list_bucket_tree(
            self.bucket, prefix=prefix, recursive=True
        ):
            if getattr(entry, "type", None) != "file" or not entry.path.endswith(
                ".json"
            ):
                continue
            key = entry.path[len(prefix) : -len(".json")]
            try:
                owner = self._owner_record(key)
            except Exception:  # an unreadable sidecar is skipped, not fatal
                logger.debug("file gc: unreadable owner record %r", key)
                continue
            if owner is None:
                continue
            records.append(
                FileRecord(
                    key=key,
                    owner=owner.get("owner"),
                    session_hash=owner.get("session_hash"),
                    committed_at=owner.get("committed_at", 0.0),
                    size=owner.get("size", 0),
                )
            )
        return records


def resolve_file_store(
    spec: str | None = None,
    **backend_kwargs: Any,
) -> FileStore:
    """Resolve the configured file store.

    Precedence mirrors the session store: an explicit argument, then
    ``GRADIO_FILE_STORE``, then the in-process default. An unknown value falls
    back to the default rather than failing.
    """
    name = (spec or os.getenv(FILE_STORE_ENV_VAR) or "inprocess").strip().lower()
    if name in ("hf", "hf-bucket", "hub", "bucket"):
        return HfBucketFileStore(**backend_kwargs)
    return LocalFileStore()
