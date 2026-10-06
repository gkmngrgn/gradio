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


def get_store():
    try:
        import redis
    except ImportError as err:
        raise RuntimeError(
            "This demo needs the 'redis' package: pip install 'redis>=5.0,<9.0'."
        ) from err
    from gradio.session_store import RedisSessionStore

    url = os.getenv("GRADIO_REDIS_URL", "redis://localhost:6379")
    client = redis.Redis.from_url(url, decode_responses=False)
    return RedisSessionStore(client, app_id="demo-counter")


def turn(room):
    room = (room or "lobby").strip() or "lobby"
    store = get_store()
    for _ in range(5):
        record = store.create(room, principal=None)
        record.state_data[0] = record.state_data.get(0, 0) + 1
        if store.save(record, record.version, principal=None) is True:
            return record.state_data[0], f"room={room}"
    raise gr.Error("Concurrent update, please try again.")


with gr.Blocks() as demo:
    gr.Markdown("## Shared counter (Redis-backed session store)")
    room = gr.Textbox(label="room", value="lobby")
    count = gr.Number(label="count", value=0)
    label = gr.Textbox(label="where", value="")
    btn = gr.Button("turn")
    btn.click(turn, room, [count, label], api_name="turn")

if __name__ == "__main__":
    demo.launch()
