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
