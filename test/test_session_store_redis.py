"""Tests for the external (Redis-backed) session store (U2).

These prove the cross-replica contract: two store instances sharing one backend
resolve the same session, writes are versioned so a stale writer is rejected,
creation is atomic, and a record is scoped to its principal.

The tests use ``fakeredis``, so they exercise the real Redis command semantics
(SET NX, WATCH/MULTI/EXEC, key expiry) without a live server.
"""

from __future__ import annotations

import base64
import datetime
import json

import fakeredis
import pytest

from gradio.session_store import (
    SESSION_ENVELOPE_SCHEMA,
    RedisSessionStore,
    SessionEnvelopeError,
    resolve_session_store,
)


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis(decode_responses=False)


@pytest.fixture
def two_stores(redis_client):
    """Two store instances over one backend -- the cross-replica setup."""
    a = RedisSessionStore(redis_client, app_id="app-1")
    b = RedisSessionStore(redis_client, app_id="app-1")
    return a, b


class TestCrossReplica:
    def test_value_written_on_a_is_read_on_b(self, two_stores):
        a, b = two_stores
        record = a.create("s1", principal=None)
        record.state_data = {0: "hello"}
        assert a.save(record, expected_version=record.version) is True

        seen = b.resolve("s1", principal=None)
        assert seen is not None
        assert seen.state_data[0] == "hello"

    def test_read_miss_does_not_fabricate_a_session(self, two_stores):
        a, b = two_stores
        assert b.resolve("nope", principal=None) is None
        assert b.contains("nope", principal=None) is False
        # Neither instance invented the session.
        assert a.contains("nope", principal=None) is False

    def test_delete_on_a_removes_for_b(self, two_stores):
        a, b = two_stores
        a.create("s1", principal=None)
        assert b.contains("s1", principal=None) is True
        a.delete("s1")
        assert b.contains("s1", principal=None) is False


class TestAuthorization:
    def test_other_principal_is_not_found(self, two_stores):
        a, b = two_stores
        a.create("s1", principal="alice")
        assert b.resolve("s1", principal="bob") is None
        assert b.contains("s1", principal="bob") is False

    def test_other_principal_cannot_write(self, two_stores):
        a = two_stores[0]
        record = a.create("s1", principal="alice")
        assert a.save(record, expected_version=record.version, principal="bob") is False

    def test_owner_reads_own_session(self, two_stores):
        a, b = two_stores
        a.create("s1", principal="alice")
        assert b.resolve("s1", principal="alice") is not None


class TestVersionedWrites:
    def test_stale_version_is_rejected(self, two_stores):
        a, _b = two_stores
        record = a.create("s1", principal=None)
        assert a.save(record, expected_version=0) is True
        # Version 0 is stale now.
        assert a.save(record, expected_version=0) is False

    def test_concurrent_writers_one_loses(self, two_stores):
        a, b = two_stores
        a.create("s1", principal=None)
        first = a.resolve("s1", principal=None)
        second = b.resolve("s1", principal=None)
        assert first is not None and second is not None
        first.state_data = {0: "from-a"}
        second.state_data = {0: "from-b"}
        assert a.save(first, expected_version=first.version) is True
        # b still holds the old version and must not clobber a's write.
        assert b.save(second, expected_version=second.version) is False
        assert a.resolve("s1", principal=None).state_data[0] == "from-a"

    def test_watch_error_is_a_retryable_conflict(self, two_stores, monkeypatch):
        import redis

        store = two_stores[0]
        record = store.create("s1", principal=None)
        pipe = store._client.pipeline()

        def conflicted_execute():
            raise redis.exceptions.WatchError("concurrent update")

        monkeypatch.setattr(pipe, "execute", conflicted_execute)
        monkeypatch.setattr(store._client, "pipeline", lambda: pipe)

        assert store.save(record, expected_version=record.version) is False


class TestAtomicCreation:
    def test_resolve_then_create_yields_one_session(self, two_stores):
        a, b = two_stores
        ra = a.resolve("s1", principal=None) or a.create("s1", principal=None)
        rb = b.resolve("s1", principal=None) or b.create("s1", principal=None)
        # The second call must observe the first, not overwrite it.
        assert ra.version == rb.version
        assert a.resolve("s1", principal=None) is not None

    def test_create_is_set_nx_guarded(self, two_stores):
        a = two_stores[0]
        a.create("s1", principal=None)
        created = a.create("s1", principal=None)
        # A second create on an existing key returns the existing version.
        assert created.version == 0


class TestTypedEnvelope:
    def test_bytes_datetime_and_int_keys_round_trip(self, two_stores):
        a, b = two_stores
        record = a.create("s1", principal=None)
        when = datetime.datetime(2026, 10, 4, 12, 0, 0, tzinfo=datetime.UTC)
        record.state_data = {0: b"\x00\xff\x10", 7: when, 42: {"nested": [1, 2, 3]}}
        assert a.save(record, expected_version=0) is True

        seen = b.resolve("s1", principal=None)
        assert seen is not None
        assert seen.state_data[0] == b"\x00\xff\x10"
        assert seen.state_data[7] == when
        assert seen.state_data[42] == {"nested": [1, 2, 3]}
        # Integer keys survive (they are dict keys, not stringified).
        assert set(seen.state_data) == {0, 7, 42}

    def test_unsupported_value_fails_loudly(self, two_stores):
        a, _b = two_stores
        record = a.create("s1", principal=None)

        class NotSerializable:
            pass

        record.state_data = {0: NotSerializable()}
        with pytest.raises(SessionEnvelopeError) as err:
            a.save(record, expected_version=0)
        assert "0" in str(err.value)

    def test_version_mismatch_envelope_is_rejected(self, two_stores, redis_client):
        a, _ = two_stores
        a.create("s1", principal=None)
        # Corrupt the stored schema version.
        raw = json.loads(redis_client.get(a._key("s1")))
        raw["schema"] = SESSION_ENVELOPE_SCHEMA + 999
        redis_client.set(a._key("s1"), json.dumps(raw))
        with pytest.raises(SessionEnvelopeError):
            a.resolve("s1", principal=None)


class TestEncryption:
    def test_envelope_is_unreadable_without_the_key(self, redis_client):
        key = base64.urlsafe_b64encode(b"0" * 32).decode()
        store = RedisSessionStore(redis_client, app_id="app-1", encryption_key=key)
        record = store.create("s1", principal=None)
        record.state_data = {0: "top-secret"}
        store.save(record, expected_version=0)

        raw = redis_client.get(store._key("s1"))
        assert b"top-secret" not in raw

    def test_wrong_key_cannot_decode(self, redis_client):
        right = base64.urlsafe_b64encode(b"0" * 32).decode()
        wrong = base64.urlsafe_b64encode(b"1" * 32).decode()
        writer = RedisSessionStore(redis_client, app_id="app-1", encryption_key=right)
        record = writer.create("s1", principal=None)
        record.state_data = {0: "secret"}
        writer.save(record, expected_version=0)

        reader = RedisSessionStore(redis_client, app_id="app-1", encryption_key=wrong)
        with pytest.raises(SessionEnvelopeError):
            reader.resolve("s1", principal=None)


class TestNamespace:
    def test_different_apps_do_not_collide(self, redis_client):
        a = RedisSessionStore(redis_client, app_id="app-1")
        b = RedisSessionStore(redis_client, app_id="app-2")
        a.create("s1", principal=None)
        assert b.contains("s1", principal=None) is False

    def test_key_is_namespaced(self, two_stores):
        a, _ = two_stores
        assert a.key_name("s1").startswith("gradio:sessions:app-1:")


class TestResolution:
    def test_resolve_registers_redis_backend(self, monkeypatch, redis_client):
        monkeypatch.setenv("GRADIO_SESSION_STORE", "redis")
        store = resolve_session_store(client=redis_client, app_id="app-1")
        assert isinstance(store, RedisSessionStore)


@pytest.mark.integration
def test_cross_replica_against_real_redis(real_redis_client):
    """Tier B: the same contract as fakeredis, on a real server."""
    a = RedisSessionStore(real_redis_client, app_id="integration")
    b = RedisSessionStore(real_redis_client, app_id="integration")
    record = a.create("s1", principal="alice")
    record.state_data[0] = {"n": 1}
    assert a.save(record, record.version) is True

    seen = b.resolve("s1", principal="alice")
    assert seen is not None
    assert seen.state_data[0] == {"n": 1}
    assert b.resolve("s1", principal="bob") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
