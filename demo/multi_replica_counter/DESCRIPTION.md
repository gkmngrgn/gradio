# Shared counter demo

A real Gradio app whose counter lives in the Redis-backed session store
(group 1 seam) instead of process memory. Open two browser tabs, type the
same room name in both, and turns in either tab advance the same counter.

## Scope

The app resolves the store explicitly: there is no
`launch(multi_replica=...)` on this branch yet -- the launch preset, and
the 10-replica compose rig that exercises it, arrive with group 5 on
`feat/multi-replica-session-durability`.

## Run

Start Redis with the Redis-only compose file:

```powershell
docker compose -f test/multi-replica/docker-compose.yml up -d
```

Launch the app (real Redis on `localhost:6379` by default, override with
`GRADIO_REDIS_URL`):

```powershell
python demo/multi_replica_counter/run.py
```

Open http://127.0.0.1:7860 in two tabs, enter the same room in both, and
alternate Turn clicks: the count climbs 1, 2, 3... regardless of tab.
