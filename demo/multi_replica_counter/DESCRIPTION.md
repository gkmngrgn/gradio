# Session counter demo

A genuine Gradio session counter: each tab holds its own `gr.State`,
turns advance it, refresh resets it. On this branch the seam behind it
is the in-process default, so behavior matches upstream session demos.

The point of this file is forward compatibility: unchanged, it is the
app the 10-replica rig on `feat/multi-replica-session-durability`
serves from shared storage, where the same per-tab session stays
consistent across replicas with no affinity.

## Run

```powershell
python demo/multi_replica_counter/run.py
```

Open http://127.0.0.1:7860 in two tabs: each tab counts independently
1, 2, 3... Refreshing a tab resets its counter -- per-tab sessions are
untouched by this stack by design.
