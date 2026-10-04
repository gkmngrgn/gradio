"""Tests for the opt-in multi-replica preset (U12).

The preset wires the session store, file store, signed auth, and durable queue
from one configuration, and validates credentials and the drain/lease pair at
launch. Without the configuration every seam stays at its in-process default.
"""

from __future__ import annotations

import fakeredis
import pytest

import gradio as gr
from gradio.exceptions import Error as GradioError


def _demo():
    with gr.Blocks() as demo:
        gr.State(0)
        gr.Textbox()
    return demo


class TestDefaultUnchanged:
    def test_no_configuration_leaves_seams_unset(self, monkeypatch):
        monkeypatch.delenv("GRADIO_MULTI_REPLICA", raising=False)
        demo = _demo()
        assert demo.session_store is None
        assert demo.file_store is None
        assert demo._queue.job_queue is None

    def test_empty_config_is_a_noop(self, monkeypatch):
        monkeypatch.delenv("GRADIO_MULTI_REPLICA", raising=False)
        demo = _demo()
        demo.configure_multi_replica(None)
        assert demo.session_store is None


class TestWiring:
    def _config(self, client, **overrides):
        config = {
            "session": {"backend": "redis", "client": client, "app_id": "app"},
            "files": {"backend": "hf", "bucket": "alice/files", "client": Fh()},
            "queue": {"backend": "redis", "client": client, "lease_ms": 60_000},
        }
        config.update(overrides)
        return config

    def test_configuration_wires_every_seam(self):
        from gradio.file_store import HfBucketFileStore
        from gradio.queue_store import RedisJobQueue
        from gradio.session_store import RedisSessionStore

        demo = _demo()
        client = fakeredis.FakeRedis(decode_responses=False)
        demo.configure_multi_replica(self._config(client))

        assert isinstance(demo.session_store, RedisSessionStore)
        assert isinstance(demo.file_store, HfBucketFileStore)
        assert isinstance(demo._queue.job_queue, RedisJobQueue)

    def test_drain_window_is_applied(self):
        demo = _demo()
        client = fakeredis.FakeRedis(decode_responses=False)
        demo.configure_multi_replica(self._config(client, drain_window=10))
        assert demo._queue.drain_timeout == 10


class TestValidation:
    def test_drain_window_at_or_above_lease_fails(self):
        demo = _demo()
        client = fakeredis.FakeRedis(decode_responses=False)
        with pytest.raises(GradioError, match="drain_window"):
            demo.configure_multi_replica(
                {
                    "session": {"backend": "redis", "client": client},
                    "files": {"backend": "hf", "bucket": "a/b", "client": Fh()},
                    "queue": {"backend": "redis", "client": client, "lease_ms": 5000},
                    "drain_window": 5,
                }
            )

    def test_missing_redis_url_fails(self, monkeypatch):
        monkeypatch.delenv("GRADIO_REDIS_URL", raising=False)
        demo = _demo()
        with pytest.raises(GradioError, match="session store"):
            demo.configure_multi_replica(
                {
                    "session": {"backend": "redis"},
                    "files": {"backend": "hf", "bucket": "a/b", "client": Fh()},
                }
            )

    def test_missing_file_bucket_fails(self, monkeypatch):
        monkeypatch.delenv("GRADIO_FILE_BUCKET", raising=False)
        demo = _demo()
        with pytest.raises(GradioError, match="file bucket"):
            demo.configure_multi_replica(
                {"files": {"backend": "hf"}, "session": {"backend": "inprocess"}}
            )


class TestEnvironmentOptIn:
    def test_env_json_wires_the_preset(self, monkeypatch):
        import json

        monkeypatch.delenv("GRADIO_REDIS_URL", raising=False)
        # A URL is required, so this asserts the JSON path is parsed and reaches
        # the credential check rather than being ignored.
        monkeypatch.setenv("GRADIO_MULTI_REPLICA", json.dumps({"session": {}}))
        demo = _demo()
        with pytest.raises(GradioError, match="session store"):
            demo.configure_multi_replica(None)

    def test_malformed_env_fails_clearly(self, monkeypatch):
        monkeypatch.setenv("GRADIO_MULTI_REPLICA", "{not json")
        demo = _demo()
        with pytest.raises(GradioError, match="not valid JSON"):
            demo.configure_multi_replica(None)

    def test_absent_env_is_a_noop(self, monkeypatch):
        monkeypatch.delenv("GRADIO_MULTI_REPLICA", raising=False)
        demo = _demo()
        demo.configure_multi_replica(None)
        assert demo.session_store is None


class Fh:
    """Minimal bucket client so the file seam does not need a real Hub."""

    def create_bucket(self, *a, **k):
        return None

    def batch_bucket_files(self, *a, **k):
        return None

    def download_bucket_files(self, *a, **k):
        return None

    def list_bucket_tree(self, *a, **k):
        return iter(())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
