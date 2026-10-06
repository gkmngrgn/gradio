"""State-sync check: N turns through the round-robin LB across replicas.

Every turn must increment the same session counter by exactly one, while
the serving replica ids show the turns actually hopped across replicas.

Run (from the repo root, Docker daemon up):
    docker compose -f test/multi-replica/docker-compose.demo.yml up -d --scale app=10
    python test/multi-replica/sync_check.py --turns 20
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gradio_client import Client


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--url", default="http://localhost:8000")
    args.add_argument("--turns", type=int, default=20)
    ns = args.parse_args()

    client = Client(ns.url)
    replicas = set()
    for expected in range(1, ns.turns + 1):
        out = client.predict(api_name="/turn")
        match = re.fullmatch(r"replica=(\S+) count=(\d+)", out.strip())
        assert match, f"unexpected output: {out!r}"
        replicas.add(match.group(1))
        assert int(match.group(2)) == expected, (
            f"state diverged on turn {expected}: {out!r}"
        )
        print(f"turn {expected}: {out}")

    assert len(replicas) > 1, f"all turns stuck to one replica: {replicas}"
    print(f"OK: {ns.turns} turns, counter monotonic, {len(replicas)} replicas served")


if __name__ == "__main__":
    main()
