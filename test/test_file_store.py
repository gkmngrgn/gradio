"""Tests for the shared, owner-scoped file store.

The object-store contract is exercised against the in-memory ``FakeHub`` bucket
from ``test_history`` so it needs no network, and ``LocalFileStore`` covers the
in-process implementation. The route wiring itself is covered by
``test_routes.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import gradio as gr
from gradio import route_utils
from gradio.file_store import (
    FileStore,
    HfBucketFileStore,
    LocalFileStore,
    resolve_file_store,
)
from test.test_history import FakeHub


def _demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Textbox()
    return demo


class TestLocalFileStore:
    def test_owner_can_read_and_another_cannot(self, tmp_path):
        store = LocalFileStore()
        src = tmp_path / "f.txt"
        src.write_text("hello")

        store.put(str(src), "k", owner="alice")

        assert store.materialize("k", "alice") == str(src)
        assert store.materialize("k", "bob") is None
        assert store.resolve("k", "bob") is None

    def test_missing_key_is_a_miss(self):
        assert LocalFileStore().resolve("nope", None) is None


class TestHfBucketFileStore:
    def test_cross_instance_and_ownership(self, tmp_path):
        hub = FakeHub()
        a = HfBucketFileStore("alice/files", app_id="app", client=hub)
        b = HfBucketFileStore("alice/files", app_id="app", client=hub)

        src = tmp_path / "img.txt"
        src.write_text("hello")
        record = a.put(str(src), "sha/img.txt", owner="alice", session_hash="s1")
        assert record.size == 5

        local = b.materialize("sha/img.txt", "alice")
        assert local is not None
        with open(local) as fh:
            assert fh.read() == "hello"

        assert b.resolve("sha/img.txt", "bob") is None
        assert b.materialize("sha/img.txt", "bob") is None

    def test_implements_the_protocol(self):
        assert isinstance(HfBucketFileStore("b", client=FakeHub()), FileStore)


class TestResolution:
    def test_default_is_local(self, monkeypatch):
        monkeypatch.delenv("GRADIO_FILE_STORE", raising=False)
        assert isinstance(resolve_file_store(), LocalFileStore)
        assert isinstance(resolve_file_store(), FileStore)

    def test_named_backend_uses_hf_buckets(self, monkeypatch):
        monkeypatch.setenv("GRADIO_FILE_STORE", "hf")
        store = resolve_file_store(bucket="alice/files", client=FakeHub())
        assert isinstance(store, HfBucketFileStore)

    def test_unknown_name_falls_back(self, monkeypatch):
        monkeypatch.setenv("GRADIO_FILE_STORE", "nope")
        assert isinstance(resolve_file_store(), LocalFileStore)


class TestBlocksWiring:
    def test_default_blocks_does_not_touch_a_store(self):
        demo = _demo()
        assert demo.file_store is None
        assert demo.store_upload("p", "k", owner="alice") is None
        assert demo.fetch_file("k", "alice") is None

    def test_upload_commits_before_fetch_and_checks_owner(self, tmp_path):
        demo = _demo()
        demo.file_store = HfBucketFileStore(
            "alice/files", app_id="app", client=FakeHub()
        )
        src = tmp_path / "u.txt"
        src.write_text("data")

        demo.store_upload(str(src), "sha/u.txt", owner="alice")

        owned = demo.fetch_file("sha/u.txt", "alice")
        assert owned is not None
        with open(owned) as fh:
            assert fh.read() == "data"
        assert demo.fetch_file("sha/u.txt", "bob") is None

    def test_generated_output_is_committed_before_its_url_is_returned(self):
        demo = _demo()
        hub = FakeHub()
        demo.file_store = HfBucketFileStore("alice/files", app_id="app", client=hub)
        generated = Path(demo.GRADIO_CACHE) / "generated.txt"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text("generated")
        data = [
            {
                "path": str(generated),
                "url": "/gradio_api/file=generated.txt",
                "meta": {"_type": "gradio.FileData"},
            }
        ]

        route_utils.store_generated_files(demo, data, "s1", "alice")

        key = route_utils.upload_store_key(str(generated), demo.GRADIO_CACHE)
        stored = demo.file_store.resolve(key, "alice")
        assert stored is not None
        assert stored.session_hash == "s1"


def test_upload_store_key_is_relative_to_upload_dir(tmp_path):
    dest = tmp_path / "sha" / "a.txt"
    key = route_utils.upload_store_key(str(dest), str(tmp_path))
    assert key == os.path.join("sha", "a.txt")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
