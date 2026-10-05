"""Tests for the durable job queue seam (U9a).

The Redis Streams contract is exercised with fakeredis in CI and, when
``GRADIO_TEST_REDIS_URL`` is set, against a real server. Both prove the same
things: a published job is delivered once, a job left pending is reclaimed from
an idle consumer, and acking removes it from the pending list.
"""

from __future__ import annotations

import asyncio
import time

import fakeredis
import pytest

from gradio.queue_store import (
    JobEnvelope,
    RedisJobQueue,
    decode_job,
    encode_job,
    resolve_job_queue,
)


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis(decode_responses=False)


def _queue(client, consumer):
    return RedisJobQueue(client, stream="jobs", group="g", consumer=consumer)


class TestEnvelope:
    def test_round_trip_preserves_fields(self):
        job = JobEnvelope(
            fn_index=2,
            inputs=[1, {"a": 2}],
            session_hash="h",
            principal="alice",
            batch=True,
            event_id="e1",
            idempotency_key="key-1",
        )
        restored = decode_job(encode_job(job))
        assert restored.to_dict() == job.to_dict()

    def test_from_dict_defaults_an_idempotency_key(self):
        job = JobEnvelope.from_dict({"fn_index": 0, "inputs": []})
        assert job.idempotency_key


class TestRedisJobQueue:
    def test_publish_is_delivered_once(self, redis_client):
        producer = _queue(redis_client, "p")
        worker = _queue(redis_client, "w")
        message_id = producer.publish(JobEnvelope(fn_index=1, inputs=["x"]))

        messages = producer.reclaim(min_idle_ms=0)
        assert [m.id for m in messages] == [message_id]
        assert messages[0].job.inputs == ["x"]
        # The published job is already pending under its producer; new-job
        # reads are reserved for the producer path, not survivor consumers.
        assert worker.read(block_ms=10) == []
        assert len(worker) == 1

    def test_reclaim_picks_up_an_idle_consumers_job(self, redis_client):
        producer = _queue(redis_client, "p")
        worker = _queue(redis_client, "w")
        producer.publish(JobEnvelope(fn_index=1, inputs=[], session_hash="s"))

        (claimed,) = producer.reclaim(min_idle_ms=0)

        # Another replica reclaims work the first consumer left pending.
        survivor = _queue(redis_client, "w2")
        reclaimed = survivor.reclaim(min_idle_ms=0)
        assert [m.id for m in reclaimed] == [claimed.id]
        assert reclaimed[0].job.session_hash == "s"

    def test_ack_removes_from_the_pending_list(self, redis_client):
        producer = _queue(redis_client, "p")
        worker = _queue(redis_client, "w")
        producer.publish(JobEnvelope(fn_index=1, inputs=[]))
        (claimed,) = producer.reclaim(min_idle_ms=0)

        worker.ack(claimed.id)
        assert len(worker) == 0
        assert worker.reclaim(min_idle_ms=0) == []

    def test_renew_keeps_the_job_out_of_reclaim(self, redis_client):
        producer = _queue(redis_client, "p")
        worker = _queue(redis_client, "w")
        producer.publish(JobEnvelope(fn_index=1, inputs=[]))
        (claimed,) = producer.reclaim(min_idle_ms=0)

        worker.renew(claimed.id)
        # The entry is still pending for this consumer (not lost, not acked).
        assert len(worker) == 1


class TestResolution:
    def test_default_is_no_external_queue(self, monkeypatch):
        monkeypatch.delenv("GRADIO_JOB_QUEUE", raising=False)
        assert resolve_job_queue() is None

    def test_named_backend_needs_a_client(self, monkeypatch):
        monkeypatch.setenv("GRADIO_JOB_QUEUE", "redis")
        with pytest.raises(RuntimeError):
            resolve_job_queue()

    def test_redis_backend_resolves(self, monkeypatch, redis_client):
        monkeypatch.setenv("GRADIO_JOB_QUEUE", "redis")
        queue = resolve_job_queue(client=redis_client)
        assert isinstance(queue, RedisJobQueue)


class TestDispatchWiring:
    def test_default_queue_does_not_publish(self):
        import gradio as gr

        with gr.Blocks() as demo:
            gr.Button()
        assert demo._queue.job_queue is None
        assert demo._queue.durable_jobs() is None

    def test_push_publishes_a_job_with_the_caller_principal(self, redis_client):
        import gradio as gr
        from gradio.data_classes import PredictBodyInternal

        with gr.Blocks() as demo:
            state = gr.State(0)
            btn = gr.Button()
            btn.click(lambda s: s, [state], None)
        queue = demo._queue
        queue.job_queue = _queue(redis_client, "producer")
        fn = demo.fns[0]

        body = PredictBodyInternal(
            data=[None], fn_index=fn._id, session_hash="s1", request=None
        )
        request = gr.Request(request=None)
        ok, _event_id, status = asyncio.run(queue.push(body, request, username="alice"))
        assert ok is True and status == "success"

        messages = queue.job_queue.reclaim(min_idle_ms=0)
        assert len(messages) == 1
        job = messages[0].job
        assert job.session_hash == "s1"
        assert job.principal == "alice"
        assert job.fn_index == fn._id

    def test_local_execution_acks_its_durable_copy_once(self, redis_client):
        import gradio as gr
        from fastapi.testclient import TestClient

        calls = []
        with gr.Blocks() as demo:
            state = gr.State(0)

            def increment(value):
                calls.append(value)
                return value + 1

            gr.Button().click(increment, [state], [state])
        from gradio.routes import App

        app = App.create_app(demo)
        queue = demo._queue
        queue.job_queue = _queue(redis_client, "producer")
        with TestClient(app) as client:
            client.get("/gradio_api/startup-events")
            response = client.post(
                "/gradio_api/queue/join",
                json={"data": [0], "fn_index": 0, "session_hash": "s1"},
            )
            assert response.status_code == 200
            for _ in range(100):
                if calls and len(queue.job_queue) == 0:
                    break
                time.sleep(0.02)

        assert calls == [0]
        assert len(queue.job_queue) == 0
        assert len(queue._applied_job_keys) == 1


class TestRedeliveryConsumer:
    """U9c: a survivor reconstructs and runs a claimed job, save-before-ack."""

    @staticmethod
    def _app():
        import gradio as gr

        with gr.Blocks() as demo:
            seen = gr.State(0)
            out = gr.Number()
            gr.Button().click(lambda s: (s + 1, s + 1), [seen], [out, seen])
        return demo

    def test_run_durable_job_applies_state_then_acks(self, redis_client):
        from gradio.queue_store import JobEnvelope

        demo = self._app()
        demo.launch(prevent_thread_lock=True)
        try:
            queue = demo._queue
            queue.job_queue = _queue(redis_client, "survivor")
            fn = demo.fns[0]
            producer = _queue(redis_client, "producer")
            producer.publish(
                JobEnvelope(fn_index=fn._id, inputs=[None], session_hash="s1")
            )
            (claimed,) = producer.reclaim(min_idle_ms=0)

            assert asyncio.run(queue.run_durable_job(claimed)) is True

            # Acked only after the work applied it.
            assert len(queue.job_queue) == 0
            stored = demo.get_session_state("s1", create=False)
            assert stored is not None
            assert stored.state_data[fn.inputs[0]._id] == 1
        finally:
            demo.close()

    def test_redelivered_job_is_deduped_by_idempotency_key(self, redis_client):
        from gradio.queue_store import JobEnvelope

        demo = self._app()
        demo.launch(prevent_thread_lock=True)
        try:
            queue = demo._queue
            queue.job_queue = _queue(redis_client, "survivor")
            fn = demo.fns[0]
            producer = _queue(redis_client, "producer")
            producer.publish(
                JobEnvelope(
                    fn_index=fn._id,
                    inputs=[None],
                    session_hash="s1",
                    idempotency_key="event-1",
                )
            )
            (claimed,) = producer.reclaim(min_idle_ms=0)

            assert asyncio.run(queue.run_durable_job(claimed)) is True
            # The same job redelivered (same key) is acked without re-running, so
            # state is applied once.
            assert asyncio.run(queue.run_durable_job(claimed)) is False
            assert "event-1" in queue._applied_job_keys
            stored = demo.get_session_state("s1", create=False)
            assert stored.state_data[fn.inputs[0]._id] == 1
        finally:
            demo.close()

    def test_unknown_function_is_left_for_a_replica_that_has_it(self, redis_client):
        from gradio.queue_store import JobEnvelope

        demo = self._app()
        demo.launch(prevent_thread_lock=True)
        try:
            queue = demo._queue
            queue.job_queue = _queue(redis_client, "survivor")
            producer = _queue(redis_client, "producer")
            producer.publish(JobEnvelope(fn_index=9999, inputs=[], session_hash="s1"))
            (claimed,) = producer.reclaim(min_idle_ms=0)

            assert asyncio.run(queue.run_durable_job(claimed)) is False
        finally:
            demo.close()

    def test_durable_consumer_is_tracked_and_exits_on_shutdown(self):
        import gradio as gr

        class EmptyQueue:
            lease_ms = 10

            def reclaim(self, min_idle_ms, count=10):
                return []

        with gr.Blocks() as demo:
            gr.Button()
        queue = demo._queue
        queue.job_queue = EmptyQueue()

        async def check():
            queue.start_durable_consumer()
            task = queue._durable_task
            assert task is not None
            assert task in queue._asyncio_tasks
            queue.stopped = True
            await asyncio.wait_for(task, timeout=2)
            assert task not in queue._asyncio_tasks
            assert queue._durable_task is None

        asyncio.run(check())


@pytest.mark.integration
def test_redelivery_against_real_redis(real_redis_client):
    """Tier B: the same reclaim/ack contract on a real server."""
    producer = RedisJobQueue(real_redis_client, stream="it:jobs", group="g")
    worker = RedisJobQueue(real_redis_client, stream="it:jobs", group="g")
    producer.publish(JobEnvelope(fn_index=1, inputs=[], session_hash="s"))
    (claimed,) = producer.reclaim(min_idle_ms=0)

    survivor = RedisJobQueue(real_redis_client, stream="it:jobs", group="g")
    reclaimed = survivor.reclaim(min_idle_ms=0)
    assert claimed.id in [m.id for m in reclaimed]

    survivor.ack(claimed.id)
    assert survivor.reclaim(min_idle_ms=0) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
