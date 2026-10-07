"""Session-state counter demo (group 1: store seam).

A genuine Gradio session counter: each tab holds its own `gr.State`,
turns advance it, refresh resets it. On this branch the seam behind it
is the in-process default, so behavior matches every upstream session
demo. The point of this file is forward compatibility: unchanged, it is
the app the 10-replica rig on `feat/multi-replica-session-durability`
serves from shared storage, where the same per-tab session stays
consistent across replicas with no affinity.
"""

import gradio as gr


def turn(counter, request: gr.Request):
    counter = (counter or 0) + 1
    return counter, f"count={counter} session={request.session_hash}"


def show(request: gr.Request):
    return 0, f"session={request.session_hash} (fresh page load)"


with gr.Blocks() as demo:
    gr.Markdown("## Session counter (one per tab)")
    state = gr.State(0)
    out = gr.Textbox(label="last turn")
    btn = gr.Button("turn")
    btn.click(turn, state, [state, out], api_name="turn")
    demo.load(show, None, [state, out])

if __name__ == "__main__":
    demo.launch()
