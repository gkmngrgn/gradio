"""Pluggable session-state storage for multi-replica Gradio apps.

Gradio normally keeps a session's state in one process (``StateHolder``). That
makes a session unreachable from any other replica, so apps behind a load
balancer without session affinity fail with "session not found" or silently see
empty state. This module defines the seam that lets session state live in an
external store, while the in-process holder stays the default.

The interface deliberately separates three operations the existing holder
conflates:

- ``contains`` -- an existence check that never creates state.
- ``resolve`` -- a read that returns ``None`` on a miss instead of minting.
- ``create`` -- the explicit atomic create-if-absent at the backend boundary.

Every method carries the caller's ``principal``. When the app sets ``auth=`` the
principal is the authenticated username; otherwise it is ``None`` and the
session hash is the only identity. A record owned by one principal is not
readable or writable by another.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from gradio.state_holder import StateHolder

if TYPE_CHECKING:
    from gradio.blocks import Blocks

SESSION_STORE_ENV_VAR = "GRADIO_SESSION_STORE"
SESSION_STORE_URL_ENV_VAR = "GRADIO_SESSION_STORE_URL"


@dataclass
class SessionRecord:
    """A session's persisted values, plus the bookkeeping a store needs.

    Only ``state_data`` and ``is_closed`` cross a process boundary; the
    component graph (``blocks_config``, ``config_values``) is reconstructed from
    the receiving replica's own app and is never persisted.
    """

    session_hash: str
    principal: str | None = None
    state_data: dict[int, Any] = field(default_factory=dict)
    is_closed: bool = False
    version: int = 0

    @property
    def owner(self) -> str | None:
        return self.principal


@runtime_checkable
class SessionStore(Protocol):
    """The session-storage seam. One implementation per backend."""

    capacity: int

    def contains(self, session_hash: str, principal: str | None) -> bool:
        """Whether the session exists and belongs to ``principal``."""

    def resolve(self, session_hash: str, principal: str | None) -> SessionRecord | None:
        """Return the session, or ``None`` on a miss. Never creates."""

    def create(self, session_hash: str, principal: str | None) -> SessionRecord:
        """Create a session. Overwrites any existing record for the key."""

    def save(
        self,
        record: SessionRecord,
        expected_version: int,
        principal: str | None = None,
    ) -> bool:
        """Persist ``record`` if its stored version still equals
        ``expected_version``. Returns ``False`` on a version mismatch or an
        ownership mismatch, so the caller can retry."""

    def delete(self, session_hash: str, principal: str | None = None) -> None:
        """Remove a session."""

    def delete_all_expired_state(self) -> None:
        """Run the expiry sweep."""

    def __len__(self) -> int:
        """Number of stored sessions (used for capacity eviction)."""

    def __contains__(self, session_hash: str) -> bool:
        """Membership by hash, without an ownership check."""


class InProcessSessionStore:
    """The default store: wraps ``StateHolder`` with no behavior change.

    This exists so the seam's callers have one interface while the default app
    keeps today's in-process semantics, including ``__getitem__``'s auto-create.
    """

    def __init__(self, blocks: Blocks | None = None, holder: StateHolder | None = None):
        self._holder = holder if holder is not None else StateHolder()
        if blocks is not None:
            self._holder.set_blocks(blocks)
        self._principals: dict[str, str | None] = {}
        self._versions: dict[str, int] = {}

    @classmethod
    def from_holder(cls, holder: StateHolder) -> InProcessSessionStore:
        return cls(holder=holder)

    @property
    def holder(self) -> StateHolder:
        return self._holder

    @property
    def capacity(self) -> int:
        return self._holder.capacity

    @capacity.setter
    def capacity(self, value: int) -> None:
        self._holder.capacity = value

    def _owns(self, session_hash: str, principal: str | None) -> bool:
        return self._principals.get(session_hash, principal) == principal

    def contains(self, session_hash: str, principal: str | None = None) -> bool:
        if session_hash not in self._holder:
            return False
        return self._owns(session_hash, principal)

    def resolve(
        self, session_hash: str, principal: str | None = None
    ) -> SessionRecord | None:
        # A no-create read: check existence first so a miss stays a miss.
        if not self.contains(session_hash, principal):
            return None
        return self._to_record(session_hash, principal)

    def create(self, session_hash: str, principal: str | None = None) -> SessionRecord:
        # Use the holder's own creation path so defaults match exactly.
        self._holder[session_hash]
        self._principals[session_hash] = principal
        self._versions[session_hash] = 0
        return self._to_record(session_hash, principal)

    def save(
        self,
        record: SessionRecord,
        expected_version: int,
        principal: str | None = None,
    ) -> bool:
        if not self._owns(record.session_hash, principal):
            return False
        state = self._holder.session_data.get(record.session_hash)
        if state is None:
            return False
        # The store owns the version, not the caller's record: a save succeeds
        # only when the caller's expectation matches what the store holds.
        if self._versions.get(record.session_hash, 0) != expected_version:
            return False
        state.state_data.clear()
        state.state_data.update(record.state_data)
        state.is_closed = record.is_closed
        self._versions[record.session_hash] = expected_version + 1
        record.version = expected_version + 1
        return True

    def delete(self, session_hash: str, principal: str | None = None) -> None:
        if not self._owns(session_hash, principal):
            return
        self._holder.delete_state(session_hash)
        self._holder.session_data.pop(session_hash, None)
        self._holder.time_last_used.pop(session_hash, None)
        self._principals.pop(session_hash, None)
        self._versions.pop(session_hash, None)

    def delete_all_expired_state(self) -> None:
        self._holder.delete_all_expired_state()

    def _to_record(self, session_hash: str, principal: str | None) -> SessionRecord:
        state = self._holder.session_data[session_hash]
        return SessionRecord(
            session_hash=session_hash,
            principal=principal,
            state_data=dict(state.state_data),
            is_closed=state.is_closed,
            version=self._versions.get(session_hash, 0),
        )

    def __len__(self) -> int:
        return len(self._holder.session_data)

    def __contains__(self, session_hash: str) -> bool:
        return session_hash in self._holder


_BUILTIN_STORES: dict[str, type] = {
    "inprocess": InProcessSessionStore,
    "in-process": InProcessSessionStore,
    "local": InProcessSessionStore,
}


def resolve_session_store(
    spec: str | None = None,
    *,
    blocks: Blocks | None = None,
    holder: StateHolder | None = None,
    **backend_kwargs: Any,
) -> InProcessSessionStore:
    """Resolve the configured store.

    Precedence, mirroring the repo's adapter convention: an explicit argument,
    then ``GRADIO_SESSION_STORE_URL``, then ``GRADIO_SESSION_STORE``, then the
    in-process default. A URL value (explicit or via the URL variable) selects
    the backend from its scheme and builds the client automatically, so one
    variable is the whole configuration. An unrecognized value falls back to
    the default rather than failing, so a typo cannot take an app down.

    ``backend_kwargs`` are forwarded to the resolved backend. External backends
    are imported lazily, so the default path never touches their dependencies.
    """
    name = (
        spec
        or os.getenv(SESSION_STORE_URL_ENV_VAR)
        or os.getenv(SESSION_STORE_ENV_VAR)
        or "inprocess"
    ).strip()

    if "://" in name:
        from urllib.parse import urlsplit

        if urlsplit(name).scheme.lower() not in ("redis", "rediss", "unix"):
            name = "inprocess"
        else:
            import redis  # lazy: the default install never depends on it

            backend_kwargs.setdefault(
                "client", redis.Redis.from_url(name, decode_responses=False)
            )
            name = "redis"

    name = name.lower()

    if name in ("redis", "redis-session"):
        client = backend_kwargs.pop("client", None) or backend_kwargs.pop("redis", None)
        if client is None:
            raise RuntimeError(
                "The Redis session store needs a client. Pass `client=`, a "
                "`redis://` URL, or set GRADIO_SESSION_STORE_URL."
            )
        return RedisSessionStore(client, **backend_kwargs)  # type: ignore[return-value]

    store_cls = _BUILTIN_STORES.get(name)
    if store_cls is None:
        # An unrecognized name falls back to the default rather than failing.
        store_cls = InProcessSessionStore

    if holder is not None:
        return InProcessSessionStore.from_holder(holder)
    if blocks is not None:
        return store_cls(blocks=blocks)  # type: ignore[call-arg]
    return store_cls()  # type: ignore[call-arg]


def register_session_store(name: str, store_cls: type) -> None:
    """Register an external backend under a name, for ``GRADIO_SESSION_STORE``."""
    _BUILTIN_STORES[name.strip().lower()] = store_cls


# ---------------------------------------------------------------------------
# Typed envelope
#
# Only JSON-representable values cross an external store. Types JSON cannot
# express natively are tagged so they survive a round trip. A value outside the
# supported set raises rather than being silently stringified (R14).
# ---------------------------------------------------------------------------

SESSION_ENVELOPE_SCHEMA = 1

# Tag used to mark a non-JSON-native scalar inside the envelope.
_TAG = "__gradio_type__"
_TAG_BYTES = "bytes"
_TAG_DATETIME = "datetime"
_TAG_DATE = "date"
_TAG_TIME = "time"
_TAG_SET = "set"
_TAG_TUPLE = "tuple"
_TAG_DECIMAL = "decimal"
_TAG_INT_KEY = "intkey"


class SessionEnvelopeError(Exception):
    """Raised when a session value cannot be encoded, or a stored envelope
    cannot be decoded (unknown schema, wrong key, or corrupt payload)."""


def _encode_value(value: Any, path: str) -> Any:
    from decimal import Decimal

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {_TAG: _TAG_BYTES, "v": base64.b64encode(value).decode("ascii")}
    if isinstance(value, datetime.datetime):
        return {_TAG: _TAG_DATETIME, "v": value.isoformat()}
    if isinstance(value, datetime.date):
        return {_TAG: _TAG_DATE, "v": value.isoformat()}
    if isinstance(value, datetime.time):
        return {_TAG: _TAG_TIME, "v": value.isoformat()}
    if isinstance(value, Decimal):
        return {_TAG: _TAG_DECIMAL, "v": str(value)}
    if isinstance(value, (set, frozenset)):
        return {
            _TAG: _TAG_SET,
            "v": [_encode_value(v, path) for v in sorted(value, key=repr)],
        }
    if isinstance(value, tuple):
        return {_TAG: _TAG_TUPLE, "v": [_encode_value(v, path) for v in value]}
    if isinstance(value, list):
        return [_encode_value(v, path) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str):
                out[k] = _encode_value(v, path)
            elif isinstance(k, int) and not isinstance(k, bool):
                out[f"{_TAG_INT_KEY}:{k}"] = _encode_value(v, path)
            else:
                raise SessionEnvelopeError(
                    f"Cannot store a dict with a {type(k).__name__} key at "
                    f"{path}. Use string or integer keys."
                )
        return out
    raise SessionEnvelopeError(
        f"Cannot store a value of type {type(value).__name__} at {path}. "
        "Session state must be JSON-representable (values that relied on silent "
        "stringification are no longer accepted)."
    )


def _decode_value(value: Any, path: str) -> Any:
    from decimal import Decimal

    if isinstance(value, list):
        return [_decode_value(v, path) for v in value]
    if isinstance(value, dict):
        if _TAG in value and len(value) == 2 and "v" in value:
            tag = value[_TAG]
            raw = value["v"]
            if tag == _TAG_BYTES:
                return base64.b64decode(raw)
            if tag == _TAG_DATETIME:
                return datetime.datetime.fromisoformat(raw)
            if tag == _TAG_DATE:
                return datetime.date.fromisoformat(raw)
            if tag == _TAG_TIME:
                return datetime.time.fromisoformat(raw)
            if tag == _TAG_DECIMAL:
                return Decimal(raw)
            if tag == _TAG_SET:
                return {_decode_value(v, path) for v in raw}
            if tag == _TAG_TUPLE:
                return tuple(_decode_value(v, path) for v in raw)
            raise SessionEnvelopeError(f"Unknown type tag {tag!r} at {path}.")
        out: dict[Any, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k.startswith(f"{_TAG_INT_KEY}:"):
                out[int(k.split(":", 1)[1])] = _decode_value(v, path)
            else:
                out[k] = _decode_value(v, path)
        return out
    return value


def encode_envelope(record: SessionRecord, closed_at: str | None = None) -> bytes:
    """Serialize a record to bytes for external storage."""
    payload = {
        "schema": SESSION_ENVELOPE_SCHEMA,
        "session_hash": record.session_hash,
        "principal": record.principal,
        "version": record.version,
        "is_closed": record.is_closed,
        "closed_at": closed_at,
        "state_data": {
            str(k): _encode_value(v, f"state_data[{k}]")
            for k, v in record.state_data.items()
        },
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_envelope(raw: bytes) -> tuple[SessionRecord, str | None]:
    """Deserialize a stored envelope, rejecting unknown schemas."""
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError) as err:
        raise SessionEnvelopeError(
            f"Stored session envelope is not valid JSON: {err}"
        ) from err
    schema = payload.get("schema")
    if schema != SESSION_ENVELOPE_SCHEMA:
        raise SessionEnvelopeError(
            f"Stored session envelope has schema {schema!r}, expected "
            f"{SESSION_ENVELOPE_SCHEMA}. It cannot be decoded by this version."
        )
    record = SessionRecord(
        session_hash=payload["session_hash"],
        principal=payload.get("principal"),
        state_data={
            int(k): _decode_value(v, f"state_data[{k}]")
            for k, v in payload.get("state_data", {}).items()
        },
        is_closed=bool(payload.get("is_closed", False)),
        version=int(payload.get("version", 0)),
    )
    return record, payload.get("closed_at")


class RedisSessionStore:
    """A session store backed by Redis, so state resolves across replicas.

    Requires a Redis client. The client is injected rather than imported, so the
    default install never depends on ``redis``. Conditional writes use
    ``WATCH``/``MULTI``/``EXEC``; creation uses ``SET ... NX``.

    When ``encryption_key`` is supplied the envelope is encrypted at rest with
    AES-GCM (via ``cryptography``), so the stored bytes are unreadable without
    the key.
    """

    def __init__(
        self,
        client: Any,
        *,
        app_id: str = "gradio",
        tenant: str = "default",
        encryption_key: str | bytes | None = None,
        prefix: str = "gradio:sessions",
        capacity: int = 10000,
        ttl_seconds: int | None = None,
    ):
        self._client = client
        self.app_id = app_id
        self.tenant = tenant
        self.prefix = prefix
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._key_bytes = getattr(client, "decode_responses", False) is False
        self._keyring = _build_keyring(encryption_key)

    def _key(self, session_hash: str) -> str | bytes:
        key = f"{self.prefix}:{self.app_id}:{self.tenant}:{session_hash}"
        return key.encode() if self._key_bytes else key

    def key_name(self, session_hash: str) -> str:
        """The key as text, for tests and diagnostics."""
        return f"{self.prefix}:{self.app_id}:{self.tenant}:{session_hash}"

    def _seal(self, data: bytes) -> bytes:
        return data if self._keyring is None else self._keyring.seal(data)

    def _open(self, data: bytes) -> bytes:
        return data if self._keyring is None else self._keyring.open(data)

    def _owns(self, session_hash: str, principal: str | None) -> bool:
        record = self._read_raw(session_hash)
        if record is None:
            return False
        return record.principal == principal

    def _read_raw(self, session_hash: str) -> SessionRecord | None:
        raw = self._client.get(self._key(session_hash))
        if raw is None:
            return None
        record, _ = decode_envelope(self._open(raw))
        return record

    def contains(self, session_hash: str, principal: str | None = None) -> bool:
        record = self._read_raw(session_hash)
        return record is not None and record.principal == principal

    def resolve(
        self, session_hash: str, principal: str | None = None
    ) -> SessionRecord | None:
        record = self._read_raw(session_hash)
        if record is None or record.principal != principal:
            return None
        self._touch(session_hash)
        return record

    def create(self, session_hash: str, principal: str | None = None) -> SessionRecord:
        record = SessionRecord(
            session_hash=session_hash, principal=principal, version=0
        )
        # SET NX: the first writer wins, so concurrent first requests cannot
        # each install a different set of defaults.
        stored = self._seal(encode_envelope(record))
        was_set = self._client.set(self._key(session_hash), stored, nx=True)
        if not was_set:
            existing = self._read_raw(session_hash)
            if existing is not None:
                return existing
        self._touch(session_hash)
        return record

    def save(
        self,
        record: SessionRecord,
        expected_version: int,
        principal: str | None = None,
    ) -> bool:
        key = self._key(record.session_hash)
        with self._client.pipeline() as pipe:
            try:
                pipe.watch(key)
                raw = pipe.get(key)
                if raw is None:
                    pipe.unwatch()
                    return False
                current, _ = decode_envelope(self._open(raw))
                if (
                    current.principal != principal
                    or current.version != expected_version
                ):
                    pipe.unwatch()
                    return False
                record.version = expected_version + 1
                pipe.multi()
                pipe.set(key, self._seal(encode_envelope(record)))
                if self.ttl_seconds:
                    pipe.expire(key, self.ttl_seconds)
                pipe.execute()
                return True
            except Exception as err:
                # A concurrent write raises WatchError: that is a version
                # conflict, not a serialization failure, so the caller retries.
                if type(err).__name__ == "WatchError":
                    return False
                raise SessionEnvelopeError(f"Failed to save session: {err}") from err

    def delete(self, session_hash: str, principal: str | None = None) -> None:
        if principal is not None and not self._owns(session_hash, principal):
            return
        self._client.delete(self._key(session_hash))

    def delete_all_expired_state(self) -> None:
        # Expiry is Redis's own TTL; nothing to sweep here.
        return None

    def _touch(self, session_hash: str) -> None:
        if self.ttl_seconds:
            self._client.expire(self._key(session_hash), self.ttl_seconds)

    def __len__(self) -> int:
        return sum(1 for _ in self._client.scan_iter(match=self._match_pattern()))

    def _match_pattern(self) -> str | bytes:
        pattern = f"{self.prefix}:{self.app_id}:{self.tenant}:*"
        return pattern.encode() if self._key_bytes else pattern

    def __contains__(self, session_hash: str) -> bool:
        return self._read_raw(session_hash) is not None


class _Keyring:
    """AES-GCM envelope encryption. Imported lazily so the default path never
    depends on ``cryptography``."""

    def __init__(self, key: bytes):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self._aead = AESGCM(key)

    def seal(self, data: bytes) -> bytes:
        import os as _os

        nonce = _os.urandom(12)
        return nonce + self._aead.encrypt(nonce, data, None)

    def open(self, data: bytes) -> bytes:
        from cryptography.exceptions import InvalidTag

        try:
            return self._aead.decrypt(data[:12], data[12:], None)
        except (InvalidTag, ValueError) as err:
            raise SessionEnvelopeError(
                "Stored session envelope could not be decrypted. The encryption "
                "key is wrong or the payload is corrupt."
            ) from err


def _build_keyring(encryption_key: str | bytes | None) -> _Keyring | None:
    if encryption_key is None:
        return None
    if isinstance(encryption_key, str):
        try:
            key = base64.urlsafe_b64decode(encryption_key)
        except Exception as err:
            raise SessionEnvelopeError(
                "The session encryption key must be base64-encoded."
            ) from err
    else:
        key = encryption_key
    if len(key) not in (16, 24, 32):
        raise SessionEnvelopeError(
            "The session encryption key must be 16, 24, or 32 bytes."
        )
    return _Keyring(key)
