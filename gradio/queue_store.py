"""Durable job queue for multi-replica Gradio apps.

Gradio's in-process ``Queue`` holds work in memory: if the replica running a job
is scaled in before it finishes, the job is gone. This module is the seam that
lets a job survive the replica that picked it up, so a rule that owns redelivery
(a Redis Stream, or another lease-based queue) can hand it to a survivor.

A job is carried as a serializable :class:`JobEnvelope` -- the function index,
inputs, session hash and principal -- so a receiving replica rebuilds the
``BlockFunction`` and ``Event`` from its own app config. The queue's own message
id is the redelivery-stable identity: the same redelivered message keeps it, so
a late duplicate can be detected. A fencing token rides along so a superseded
worker's state write can be rejected.

Redis Streams is the bundled backend: consumer groups give at-least-once
delivery through the pending-entries list, and ``XAUTOCLAIM`` recovers a job
whose consumer went idle. With no queue configured the in-process path is
unchanged.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

JOB_QUEUE_ENV_VAR = "GRADIO_JOB_QUEUE"
JOB_FIELD = "job"


class JobEncodingError(Exception):
    """A job could not be encoded, or a stored job could not be decoded."""


@dataclass
class JobEnvelope:
    """A durable, reconstructible description of one queued function call."""

    fn_index: int
    inputs: Any
    session_hash: str | None = None
    principal: str | None = None
    batch: Any = None
    event_id: str | None = None
    # Stable across redeliveries, so a redelivered job can detect an already
    # applied write.
    idempotency_key: str = field(default_factory=lambda: secrets.token_urlsafe(16))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fn_index": self.fn_index,
            "inputs": self.inputs,
            "session_hash": self.session_hash,
            "principal": self.principal,
            "batch": self.batch,
            "event_id": self.event_id,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobEnvelope:
        return cls(
            fn_index=int(data["fn_index"]),
            inputs=data.get("inputs"),
            session_hash=data.get("session_hash"),
            principal=data.get("principal"),
            batch=data.get("batch"),
            event_id=data.get("event_id"),
            idempotency_key=data.get("idempotency_key") or secrets.token_urlsafe(16),
        )


def encode_job(job: JobEnvelope) -> bytes:
    """Serialize a job for the queue. A value the codec cannot hold fails loudly."""
    try:
        return json.dumps(job.to_dict(), separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as err:
        raise JobEncodingError(f"Job could not be encoded: {err}") from err


def decode_job(raw: bytes | str) -> JobEnvelope:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as err:
        raise JobEncodingError(f"Stored job is not valid JSON: {err}") from err
    try:
        return JobEnvelope.from_dict(data)
    except (KeyError, TypeError, ValueError) as err:
        raise JobEncodingError(f"Stored job is malformed: {err}") from err


@dataclass
class JobMessage:
    """One claimed job plus the queue identity it must be acked by."""

    id: str
    job: JobEnvelope
    consumer: str | None = None


@runtime_checkable
class JobQueue(Protocol):
    """The durable-queue seam. One implementation per backend."""

    def publish(self, job: JobEnvelope) -> str:
        """Enqueue a job; returns the queue message id (the idempotency key)."""

    def read(self, count: int = 1, block_ms: int | None = None) -> list[JobMessage]:
        """Claim new jobs for this consumer, oldest first."""

    def reclaim(self, min_idle_ms: int, count: int = 10) -> list[JobMessage]:
        """Claim jobs a consumer left pending past ``min_idle_ms``."""

    def ack(self, message_id: str) -> None:
        """Acknowledge a finished job so it is not redelivered."""

    def renew(self, message_id: str) -> None:
        """Extend the lease on a job still running."""

    def __len__(self) -> int:
        """Number of delivered-but-unacked jobs (the pending list)."""


class RedisJobQueue:
    """A durable job queue backed by a Redis Stream consumer group.

    The client is injected, so the default install never imports ``redis``.
    ``XAUTOCLAIM`` recovers work from a consumer that went idle past
    ``lease_ms``.
    """

    def __init__(
        self,
        client: Any,
        *,
        stream: str = "gradio:jobs",
        group: str = "gradio",
        consumer: str | None = None,
        lease_ms: int = 60_000,
    ):
        self._client = client
        self.stream = stream
        self.group = group
        self.consumer = consumer or f"consumer-{secrets.token_hex(4)}"
        self.lease_ms = lease_ms
        self._ensure_group()

    def _key(self, name: str) -> str | bytes:
        # Matches the client's decode setting so keys address the same entries.
        return (
            name.encode()
            if getattr(self._client, "decode_responses", False) is False
            else name
        )

    def _ensure_group(self) -> None:
        try:
            self._client.xgroup_create(
                self._key(self.stream), self.group, id="0", mkstream=True
            )
        except Exception as err:  # BUSYGROUP: the group already exists
            if "BUSYGROUP" not in str(err):
                raise

    @staticmethod
    def _field(fields: dict, name: str) -> bytes | str | None:
        if name in fields:
            return fields[name]
        encoded = name.encode()
        return fields.get(encoded)

    def publish(self, job: JobEnvelope) -> str:
        message_id = self._client.xadd(
            self._key(self.stream), {JOB_FIELD: encode_job(job)}
        )
        message_id = (
            message_id.decode() if isinstance(message_id, bytes) else message_id
        )
        # The in-process queue executes the first attempt so it can stream to
        # the submitting client. Put the durable copy in the pending list now;
        # other replicas only reclaim it after this producer's lease expires.
        self._client.xreadgroup(
            self.group,
            self.consumer,
            {self._key(self.stream): ">"},
            count=1,
            block=0,
        )
        return message_id

    def read(self, count: int = 1, block_ms: int | None = None) -> list[JobMessage]:
        streams = {self._key(self.stream): ">"}
        response = self._client.xreadgroup(
            self.group,
            self.consumer,
            streams,
            count=count,
            block=block_ms,
        )
        return self._messages_from(response, self.consumer)

    def reclaim(self, min_idle_ms: int, count: int = 10) -> list[JobMessage]:
        cursor = "0-0"
        claimed: list[JobMessage] = []
        while True:
            result = self._client.xautoclaim(
                self._key(self.stream),
                self.group,
                self.consumer,
                min_idle_time=min_idle_ms,
                start_id=cursor,
                count=count,
            )
            cursor = result[0]
            if isinstance(cursor, bytes):
                cursor = cursor.decode()
            for message_id, fields in result[1]:
                claimed.append(self._message(message_id, fields, self.consumer))
            if cursor in ("0-0", "0") or not result[1]:
                break
        return claimed

    def ack(self, message_id: str) -> None:
        self._client.xack(self._key(self.stream), self.group, message_id)

    def renew(self, message_id: str) -> None:
        # XCLAIM with no consumer change and a 0 idle time resets the pending
        # entry's idle clock, which is the lease.
        self._client.xclaim(
            self._key(self.stream),
            self.group,
            self.consumer,
            min_idle_time=0,
            message_ids=[message_id],
            idle=0,
        )

    def _messages_from(self, response, consumer: str | None) -> list[JobMessage]:
        messages: list[JobMessage] = []
        for _stream, entries in response or []:
            for message_id, fields in entries:
                messages.append(self._message(message_id, fields, consumer))
        return messages

    def _message(self, message_id, fields, consumer: str | None) -> JobMessage:
        raw = self._field(fields, JOB_FIELD)
        job = decode_job(raw if raw is not None else b"{}")
        mid = message_id.decode() if isinstance(message_id, bytes) else message_id
        return JobMessage(id=mid, job=job, consumer=consumer)

    def __len__(self) -> int:
        pending = self._client.xpending(self._key(self.stream), self.group)
        if not pending:
            return 0
        if isinstance(pending, dict):
            return int(pending.get("pending", 0))
        return int(pending[0])


def resolve_job_queue(
    spec: str | None = None,
    **backend_kwargs: Any,
) -> JobQueue | None:
    """Resolve the configured job queue.

    Precedence mirrors the other seams: an explicit argument, then
    ``GRADIO_JOB_QUEUE``, then no external queue (``None`` is returned so the
    caller keeps the in-process path).
    """
    name = (spec or os.getenv(JOB_QUEUE_ENV_VAR) or "").strip().lower()
    if name in ("redis", "redis-streams", "streams"):
        client = backend_kwargs.pop("client", None) or backend_kwargs.pop("redis", None)
        if client is None:
            raise RuntimeError(
                "The Redis job queue needs a client. Pass `client=` or set the "
                "connection so it can be created."
            )
        return RedisJobQueue(client, **backend_kwargs)
    return None  # no external queue: the in-process Queue is unchanged
