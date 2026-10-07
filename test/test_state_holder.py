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


class TestConcurrentMerge:
    """Two turns racing on one session merge per key; close sticks."""

    @staticmethod
    def _demo():
        with gr.Blocks() as demo:
            gr.State(0)
        return demo

    @staticmethod
    def _store():
        return RedisSessionStore(
            fakeredis.FakeRedis(decode_responses=False), app_id="app-1"
        )

    def _seeded(self, demo, data):
        seed = demo.get_session_state("s", principal=None, create=True)
        seed.state_data.update(data)
        assert demo.save_session_state(seed, "s", principal=None) is True

    def test_deleted_key_stays_deleted_on_conflict(self):
        demo = self._demo()
        demo.session_store = self._store()
        self._seeded(demo, {0: "a", 1: "a"})
        turn_a = demo.get_session_state("s", principal=None)
        turn_b = demo.get_session_state("s", principal=None)
        del turn_a.state_data[0]
        turn_b.state_data[1] = "b"
        assert demo.save_session_state(turn_a, "s", principal=None) is True
        assert demo.save_session_state(turn_b, "s", principal=None) is True
        final = demo.session_store.resolve("s", principal=None)
        assert final is not None
        assert 0 not in final.state_data
        assert final.state_data[1] == "b"

    def test_concurrent_close_wins_over_open_turn(self):
        demo = self._demo()
        demo.session_store = self._store()
        self._seeded(demo, {0: "a"})
        turn = demo.get_session_state("s", principal=None)
        closer = demo.get_session_state("s", principal=None)
        closer.is_closed = True
        assert demo.save_session_state(closer, "s", principal=None) is True
        turn.state_data[0] = "b"
        assert demo.save_session_state(turn, "s", principal=None) is True
        final = demo.session_store.resolve("s", principal=None)
        assert final is not None
        assert final.is_closed is True
        assert final.state_data[0] == "b"


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
