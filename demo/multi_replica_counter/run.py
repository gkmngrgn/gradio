"""10-replica state-sync demo counter.

Each replica reports its own container hostname with the count, so the
sync check can prove turns hopped across replicas while the counter
stayed monotonic. All replicas share one Redis; files go to the
operator's Hugging Face Storage Bucket (only touched on file ops).
"""

import os
import socket

import gradio as gr

REPLICA = socket.gethostname()


def turn(counter):
    counter = (counter or 0) + 1
    return counter, f"replica={REPLICA} count={counter}"


with gr.Blocks() as demo:
    gr.Markdown("## Multi-replica counter (10 replicas, no affinity)")
    state = gr.State(0)
    out = gr.Textbox(label="last turn")
    btn = gr.Button("turn", api_name="turn")
    btn.click(turn, state, [state, out])

if __name__ == "__main__":
    redis_url = os.environ["GRADIO_REDIS_URL"]
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        multi_replica={
            "session": {"url": redis_url},
            "files": {"bucket": os.environ["GRADIO_FILE_BUCKET"]},
            "queue": {"url": redis_url, "lease_ms": 60_000},
            "drain_window": 20,
        },
    )
