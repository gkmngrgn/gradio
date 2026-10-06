"""Shared counter demo for the session-store seam (group 1).

A real Gradio app whose counter lives in the Redis-backed session store
instead of process memory. Open two tabs, type the same room name in both,
and turns in either tab advance the same counter -- the store, not the
process, owns the state.

There is no `launch(multi_replica=...)` on this branch yet (it arrives
with the preset in group 5), so the app resolves the store explicitly.
Needs a Redis server (default: localhost:6379):
    docker compose -f test/multi-replica/docker-compose.yml up -d

Run:
    python demo/multi_replica_counter/run.py
"""

import os

import gradio as gr
from gradio.session_store import resolve_session_store


_store_holder: dict = {}


def get_store():
    # One process-wide instance bound to this app: the in-process backend
    # mints sessions from the app's Blocks, so it needs the binding (the
    # Redis backend ignores it -- the receiving replica rebuilds state
    # from its own app).
    if "store" not in _store_holder:
        if os.getenv("GRADIO_SESSION_STORE_URL"):
            _store_holder["store"] = resolve_session_store(app_id="demo-counter")
        else:
            _store_holder["store"] = resolve_session_store(
                blocks=demo, app_id="demo-counter"
            )
    return _store_holder["store"]


def turn(room):
    room = (room or "lobby").strip() or "lobby"
    store = get_store()
    for _ in range(5):
        record = store.create(room, principal=None)
        record.state_data[0] = record.state_data.get(0, 0) + 1
        if store.save(record, record.version, principal=None) is True:
            return record.state_data[0], f"room={room}"
    raise gr.Error("Concurrent update, please try again.")


def show(room):
    room = (room or "lobby").strip() or "lobby"
    record = get_store().resolve(room, principal=None)
    count = record.state_data.get(0, 0) if record is not None else 0
    return count, f"room={room}"


with gr.Blocks() as demo:
    gr.Markdown("## Shared counter (Redis-backed session store)")
    room = gr.Textbox(label="room", value="lobby")
    count = gr.Number(label="count", value=0)
    label = gr.Textbox(label="where", value="")
    btn = gr.Button("turn")
    btn.click(turn, room, [count, label], api_name="turn")
    demo.load(show, room, [count, label])

if __name__ == "__main__":
    demo.launch()
