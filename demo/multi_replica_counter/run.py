"""Counter demo for the session-store seam (group 1).

No `launch(multi_replica=...)` here yet -- that arrives with the preset in
group 5. This demo drives the seam API directly: two store instances stand
in for two replicas, and turns alternate between them the way a
round-robin load balancer would distribute them.

Needs a Redis server (default: localhost:6379):
    docker compose -f test/multi-replica/docker-compose.yml up -d

Run:
    python demo/multi_replica_counter/run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import redis

from gradio.session_store import RedisSessionStore


def fail(message):
    print(f"FAILED: {message}")
    raise SystemExit(1)


def turn(store, session_hash, principal, label):
    record = store.resolve(session_hash, principal=principal)
    if record is None:
        fail(f"{label} cannot see the session")
    record.state_data[0] = record.state_data.get(0, 0) + 1
    if store.save(record, record.version, principal=principal) is not True:
        fail(f"{label} lost a versioned write")
    print(f"{label}: count={record.state_data[0]}")
    return record.state_data[0]


def main() -> None:
    url = os.getenv("GRADIO_TEST_REDIS_URL", "redis://localhost:6379")
    client = redis.Redis.from_url(url, decode_responses=False)

    replica_a = RedisSessionStore(client, app_id="demo-counter")
    replica_b = RedisSessionStore(client, app_id="demo-counter")

    replica_a.delete("counter-1", principal="alice")
    replica_a.create("counter-1", principal="alice")
    pairs = [(replica_a, "replica A"), (replica_b, "replica B")] * 5
    counts = [turn(s, "counter-1", "alice", name) for s, name in pairs]
    if counts != list(range(1, 11)):
        fail(f"state diverged: {counts}")
    print("OK: 10 turns across 2 replicas, counter 1..10, no affinity")


if __name__ == "__main__":
    main()
