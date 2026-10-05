import fakeredis
import pytest

import gradio as gr
from gradio.session_store import RedisSessionStore, SessionEnvelopeError
from gradio.state_holder import StateHolder


def _holder(demo: gr.Blocks) -> StateHolder:
    holder = StateHolder()
    holder.set_blocks(demo)
    return holder


def _demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.State(0)
        gr.Textbox()
    return demo


class TestStateHolderDoesNotAccumulate:
    """See https://github.com/gradio-app/gradio/issues/11602."""

    def test_expiring_state_keeps_the_session(self):
        holder = _holder(_demo())
        holder["abc"].is_closed = True

        holder.delete_all_expired_state()

        assert "abc" in holder.session_data
        assert "abc" in holder.time_last_used

    def test_evicting_over_capacity_forgets_the_last_used_time(self):
        holder = _holder(_demo())
        holder.capacity = 2

        for i in range(5):
            holder[f"s{i}"]

        assert len(holder.session_data) == 2
        assert set(holder.time_last_used) == set(holder.session_data)


class _Opaque:
    """A value the typed envelope cannot represent."""


class TestLifecycle:
    """U11: closed sessions are retained then swept; cleanup fires on eviction."""

    @staticmethod
    def _demo(calls: list):
        import gradio as gr

        def on_delete(value):
            calls.append(value)

        with gr.Blocks() as demo:
            state = gr.State(0, delete_callback=on_delete)
        return demo, state

    @staticmethod
    def _client():
        return fakeredis.FakeRedis(decode_responses=False)

    def _store(self):
        return RedisSessionStore(self._client(), app_id="app-1")

    @staticmethod
    def _backdate_close(store, session_hash, seconds):
        """Rewrite the stored envelope's closed_at into the past."""
        import json

        from gradio.session_store import decode_envelope, encode_envelope

        record, _ = decode_envelope(
            store._read_raw(session_hash)
            and store._open(store._client.get(store._key(session_hash)))
        )
        record.closed_at -= seconds
        store._client.set(
            store._key(session_hash), store._seal(encode_envelope(record))
        )

    def test_closed_session_is_retained_then_swept(self):
        calls: list = []
        demo, state = self._demo(calls)
        demo.session_store = self._store()

        session = demo.get_session_state("h")
        session.state_data[state._id] = "keep"
        session.is_closed = True
        assert demo.save_session_state(session, "h") is True

        # Inside retention it stays and nothing is cleaned up.
        assert demo.sweep_sessions(closed_retention=3600) == 0
        assert demo.session_store.resolve("h", None) is not None

        # Past retention it is removed and the component callback fires.
        self._backdate_close(demo.session_store, "h", 7200)
        assert demo.sweep_sessions(closed_retention=3600) == 1
        assert demo.session_store.resolve("h", None) is None
        assert calls == ["keep"]

    def test_open_session_is_not_swept(self):
        demo, _ = self._demo([])
        demo.session_store = self._store()
        demo.get_session_state("h")
        assert demo.sweep_sessions(closed_retention=0) == 0
        assert demo.session_store.resolve("h", None) is not None

    def test_default_path_does_not_sweep(self):
        demo, _ = self._demo([])
        assert demo.session_store is None
        assert demo.sweep_sessions() == 0


class TestOrphanedFileCollection:
    """U11: files whose owning session is gone are collected."""

    def test_file_orphaned_by_a_swept_session_is_collected(self, tmp_path):
        from gradio.file_store import HfBucketFileStore

        import gradio as gr
        from test.test_history import FakeHub

        hub = FakeHub()
        with gr.Blocks() as demo:
            gr.Textbox()
        demo.session_store = RedisSessionStore(
            fakeredis.FakeRedis(decode_responses=False), app_id="app-1"
        )
        demo.file_store = HfBucketFileStore("alice/files", app_id="app", client=hub)

        src = tmp_path / "u.txt"
        src.write_text("data")
        demo.store_upload(str(src), "sha/u.txt", owner=None, session_hash="h")
        assert demo.file_store.resolve("sha/u.txt", None) is not None

        from gradio.route_utils import _collect_orphaned_files

        # A fresh upload may precede creation of its session record; keep it
        # during the grace period.
        _collect_orphaned_files(demo)
        assert demo.file_store.resolve("sha/u.txt", None) is not None

        # The session never existed, so the file is orphaned and removed.
        _collect_orphaned_files(demo, grace_seconds=0)
        assert demo.file_store.resolve("sha/u.txt", None) is None
        assert hub.files == {}


class TestSerializationFailure:
    """R14: a value outside the envelope fails loudly and names its source."""

    @staticmethod
    def _demo():
        with gr.Blocks() as demo:
            state = gr.State(0)
        return demo, state

    @staticmethod
    def _store():
        return RedisSessionStore(
            fakeredis.FakeRedis(decode_responses=False), app_id="app-1"
        )

    def test_label_for_names_the_state(self):
        demo, state = self._demo()
        session = _holder(demo)["s"]
        assert session.label_for(state._id) == f"state (id {state._id})"

    def test_supported_value_stores(self):
        demo, state = self._demo()
        demo.session_store = self._store()
        session = demo.get_session_state("h")
        session.state_data[state._id] = {"a": [1, 2, {"b": (3, 4)}]}

        assert demo.save_session_state(session, "h") is True
        stored = demo.session_store.resolve("h", None)
        assert stored is not None
        assert stored.state_data[state._id] == {"a": [1, 2, {"b": (3, 4)}]}

    def test_unsupported_value_names_the_state_and_type(self):
        demo, state = self._demo()
        demo.session_store = self._store()
        session = demo.get_session_state("h")
        session.state_data[state._id] = _Opaque()

        with pytest.raises(SessionEnvelopeError) as err:
            demo.save_session_state(session, "h")
        message = str(err.value)
        assert f"state (id {state._id})" in message
        assert "_Opaque" in message

    def test_unsupported_value_does_not_stringify_or_fall_back(self):
        demo, state = self._demo()
        demo.session_store = self._store()
        session = demo.get_session_state("h")
        session.state_data[state._id] = "kept"
        assert demo.save_session_state(session, "h") is True

        session.state_data[state._id] = _Opaque()
        with pytest.raises(SessionEnvelopeError):
            demo.save_session_state(session, "h")

        stored = demo.session_store.resolve("h", None)
        assert stored is not None
        assert stored.state_data[state._id] == "kept"


class TestSessionSaveFailure:
    def test_process_api_surfaces_an_exhausted_session_save(self, monkeypatch):
        import asyncio

        import gradio as gr
        from gradio.exceptions import Error
        from gradio.state_holder import SessionState

        with gr.Blocks() as demo:
            state = gr.State(0)
            gr.Button().click(lambda value: value + 1, state, state)
        demo.session_store = object()
        monkeypatch.setattr(demo, "save_session_state", lambda *args, **kwargs: False)

        with pytest.raises(Error, match="Session state could not be saved"):
            asyncio.run(
                demo.process_api(
                    0,
                    [None],
                    state=SessionState(demo),
                    session_hash="s1",
                )
            )
