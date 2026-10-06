"""Tests for the session-store adapter interface (U1).

The adapter seam lets a session be resolved through a pluggable backend so that
session state can live outside a single process. The in-process implementation is
the default and must preserve existing ``StateHolder`` behavior.
"""

from __future__ import annotations

import pytest

import gradio as gr
from gradio.session_store import (
    InProcessSessionStore,
    SessionRecord,
    SessionStore,
    resolve_session_store,
)
from gradio.state_holder import StateHolder


def _demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.State(0)
        gr.Textbox()
    return demo


def _store() -> InProcessSessionStore:
    return InProcessSessionStore(_demo())


class TestDefaultIsInProcess:
    def test_no_configuration_resolves_in_process(self, monkeypatch):
        monkeypatch.delenv("GRADIO_SESSION_STORE", raising=False)
        store = resolve_session_store()
        assert isinstance(store, InProcessSessionStore)

    def test_default_implements_the_interface(self):
        assert isinstance(_store(), SessionStore)

    def test_unknown_backend_name_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GRADIO_SESSION_STORE", "not-a-real-backend")
        store = resolve_session_store()
        assert isinstance(store, InProcessSessionStore)


class TestRoundTrip:
    def test_value_written_is_read_back(self):
        store = _store()
        created = store.create("s1", principal=None)
        created.state_data[0] = 42
        store.save(created, expected_version=created.version)

        record = store.resolve("s1", principal=None)
        assert record is not None
        assert record.state_data[0] == 42

    def test_capacity_still_evicts_least_recently_used(self):
        store = _store()
        store.capacity = 2
        for i in range(5):
            store.create(f"s{i}", principal=None)
        assert len(store) == 2

    def test_no_create_resolve_does_not_exercise_capacity(self):
        # A no-create read must not grow the store, so a miss cannot evict.
        store = _store()
        store.capacity = 1
        for i in range(5):
            store.resolve(f"s{i}", principal=None)
        assert len(store) == 0


class TestExistenceCheckDoesNotCreate:
    def test_contains_does_not_mint_a_session(self):
        store = _store()
        assert store.contains("missing", principal=None) is False
        assert "missing" not in store

    def test_resolve_missing_returns_none_without_creating(self):
        store = _store()
        assert store.resolve("missing", principal=None) is None
        assert len(store) == 0

    def test_resolve_then_create_mints_once(self):
        store = _store()
        first = store.resolve("s1", principal=None) or store.create(
            "s1", principal=None
        )
        second = store.resolve("s1", principal=None) or store.create(
            "s1", principal=None
        )
        assert first.session_hash == second.session_hash
        assert len(store) == 1


class TestAuthorization:
    def test_read_by_another_principal_is_not_found(self):
        store = _store()
        store.create("s1", principal="alice")
        assert store.resolve("s1", principal="bob") is None
        assert store.contains("s1", principal="bob") is False

    def test_write_by_another_principal_is_rejected(self):
        store = _store()
        record = store.create("s1", principal="alice")
        assert (
            store.save(record, expected_version=record.version, principal="bob")
            is False
        )

    def test_owner_can_read_own_session(self):
        store = _store()
        store.create("s1", principal="alice")
        assert store.resolve("s1", principal="alice") is not None


class TestVersionedWrites:
    def test_stale_version_is_rejected(self):
        store = _store()
        record = store.create("s1", principal=None)
        assert store.save(record, expected_version=0) is True
        # Version 0 is now stale; a second caller holding it must be rejected.
        assert store.save(record, expected_version=0) is False

    def test_matching_version_bumps_the_record(self):
        store = _store()
        record = store.create("s1", principal=None)
        assert store.save(record, expected_version=0) is True
        stored = store.resolve("s1", principal=None)
        assert stored is not None
        assert stored.version == 1


class TestRecordingWrapsExistingHolder:
    def test_record_exposes_state_and_closed_at(self):
        record = SessionRecord(session_hash="s1", principal=None)
        record.is_closed = True
        assert record.is_closed is True
        assert record.state_data == {}


class TestRetiresWithStateHolder:
    def test_store_can_be_built_from_a_holder(self):
        holder = StateHolder()
        holder.set_blocks(_demo())
        store = InProcessSessionStore.from_holder(holder)
        record = store.create("s1", principal=None)
        assert record.session_hash == "s1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
