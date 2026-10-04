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
- ``resolve_or_create`` -- the explicit create-if-absent.

Every method carries the caller's ``principal``. When the app sets ``auth=`` the
principal is the authenticated username; otherwise it is ``None`` and the
session hash is the only identity. A record owned by one principal is not
readable or writable by another.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from gradio.state_holder import StateHolder

if TYPE_CHECKING:
    from gradio.blocks import Blocks

SESSION_STORE_ENV_VAR = "GRADIO_SESSION_STORE"


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

    def resolve_or_create(
        self, session_hash: str, principal: str | None
    ) -> SessionRecord:
        """Return the session, creating it atomically if it is absent."""

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

    def resolve_or_create(
        self, session_hash: str, principal: str | None = None
    ) -> SessionRecord:
        if not self.contains(session_hash, principal):
            return self.create(session_hash, principal)
        return self.resolve(session_hash, principal)  # type: ignore[return-value]

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
) -> InProcessSessionStore:
    """Resolve the configured store.

    Precedence, mirroring the repo's adapter convention: an explicit argument,
    then ``GRADIO_SESSION_STORE``, then the in-process default. An unrecognized
    value falls back to the default rather than failing, so a typo cannot take
    an app down.

    External backend names are registered by the modules that implement them;
    until one is configured this always returns the in-process store.
    """
    name = (spec or os.getenv(SESSION_STORE_ENV_VAR) or "inprocess").strip().lower()
    store_cls = _BUILTIN_STORES.get(name)
    if store_cls is None:
        # An external backend would be looked up here. Until one is bundled the
        # only correct behavior is the in-process default.
        store_cls = InProcessSessionStore

    store = store_cls(blocks=blocks, holder=holder) if holder or blocks else store_cls()
    if holder is not None and store_cls is InProcessSessionStore:
        store = InProcessSessionStore.from_holder(holder)
    return store


def register_session_store(name: str, store_cls: type) -> None:
    """Register an external backend under a name, for ``GRADIO_SESSION_STORE``."""
    _BUILTIN_STORES[name.strip().lower()] = store_cls
