# Multi-replica counter demo

Proves the session-store seam (group 1): two store instances stand in for
two replicas behind a round-robin load balancer. Turns alternate between
them; the counter must still come out exactly 1..10 with no affinity.

## Scope

This demo drives the seam API directly. There is no
`launch(multi_replica=...)` here yet -- the launch preset arrives with
group 5 on `feat/multi-replica-session-durability`. For the full
10-container app version, see `demo/multi_replica_counter/docker-compose.yml`
on that branch.

## Run

Start Redis only -- that compose file holds a single Redis service, so
no `--scale` applies here (the 10-app scaling lives in
`docker-compose.yml` next to this demo):

```powershell
docker compose -f test/multi-replica/docker-compose.yml up -d
```

Run the demo (real Redis on `localhost:6379` by default, override with
`GRADIO_TEST_REDIS_URL`):

```powershell
python demo/multi_replica_counter/run.py
```

Expected output:

```
replica A: count=1
replica B: count=2
...
replica B: count=10
OK: 10 turns across 2 replicas, counter 1..10, no affinity
```

The demo deletes and recreates its session on every run, so reruns are
deterministic.
