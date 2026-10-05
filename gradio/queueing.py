from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import platform
import random
import time
import traceback
import uuid
from asyncio import Queue as AsyncQueue
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, cast

import fastapi
import numpy as np
from anyio.to_thread import run_sync

from gradio import route_utils, routes
from gradio.caching import CacheMissError, ProbeCache
from gradio.data_classes import (
    PredictBodyInternal,
)
from gradio.exceptions import Error
from gradio.helpers import TrackedIterable
from gradio.profiling import (
    PROFILING_ENABLED,
    RequestTrace,
    collector,
    get_current_trace,
    set_current_trace,
)
from gradio.server_messages import (
    EstimationMessage,
    EventMessage,
    LogMessage,
    ProcessCompletedMessage,
    ProcessGeneratingMessage,
    ProcessStartsMessage,
    ProgressMessage,
    ProgressUnit,
    ServerMessage,
)
from gradio.utils import (
    LRUCache,
    error_payload,
    run_coro_in_background,
    safe_aclose_iterator,
    safe_get_lock,
    set_task_name,
)

from .block_function import BlockFunction

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gradio.block_function import BlockFunction
    from gradio.blocks import Blocks


class Event:
    def __init__(
        self,
        session_hash: str | None,
        fn: BlockFunction,
        request: fastapi.Request,
        username: str | None,
    ):
        self._id = uuid.uuid4().hex
        self.session_hash: str = session_hash or self._id
        self.fn = fn
        self.request = request
        self.username = username
        self.concurrency_id = fn.concurrency_id
        self.data: PredictBodyInternal | None = None
        self.progress: ProgressMessage | None = None
        self.progress_pending: bool = False
        self.alive = True
        self.closed = False
        self.n_calls = 0
        self.run_time: float = 0
        self.enqueue_time: float = time.monotonic()
        self.signal = asyncio.Event()
        # True when the client receives this event's messages on the same
        # request that submitted it (sse_v4), instead of on the session-wide
        # `queue/data` stream.
        self.own_stream = False

    @property
    def streaming(self):
        return self.fn.connection == "stream"

    @property
    def is_finished(self):
        if not self.streaming:
            raise ValueError("Cannot access if_finished during a non-streaming event")
        if self.closed:
            return True
        if self.fn.time_limit is None:
            return False
        return self.run_time >= self.fn.time_limit


class EventQueue:
    def __init__(self, concurrency_id: str, concurrency_limit: int | None):
        self.queue: list[Event] = []
        self.concurrency_id = concurrency_id
        self.concurrency_limit = concurrency_limit
        self.current_concurrency = 0
        self.start_times_per_fn: defaultdict[BlockFunction, set[float]] = defaultdict(
            set
        )


class ProcessTime:
    def __init__(self):
        self.process_time = 0
        self.count = 0
        self.avg_time = 0

    def add(self, time: float):
        self.process_time += time
        self.count += 1
        self.avg_time = self.process_time / self.count


class Queue:
    def __init__(
        self,
        live_updates: bool,
        concurrency_count: int,
        update_intervals: float,
        max_size: int | None,
        blocks: Blocks,
        default_concurrency_limit: int | None | Literal["not_set"] = "not_set",
    ):
        self.pending_messages_per_session: LRUCache[str, AsyncQueue[EventMessage]] = (
            LRUCache(2000)
        )
        # Messages for events that stream on their own request (sse_v4). An
        # entry lives exactly as long as that request is open.
        self.pending_messages_per_event: dict[str, AsyncQueue[EventMessage]] = {}
        self.pending_event_ids_session: dict[str, set[str]] = {}
        self.event_ids_to_events: dict[str, Event] = {}
        self.pending_message_lock = safe_get_lock()
        self.event_queue_per_concurrency_id: dict[str, EventQueue] = {}
        self.stopped = False
        # Opt-in graceful drain window. None keeps the default immediate stop.
        self.drain_timeout: float | None = None
        self.max_thread_count = concurrency_count
        self.update_intervals = update_intervals
        self.active_jobs: list[None | list[Event]] = []
        self.delete_lock = safe_get_lock()
        self.server_app = None
        self.process_time_per_fn: defaultdict[BlockFunction, ProcessTime] = defaultdict(
            ProcessTime
        )
        self.live_updates = live_updates
        self.sleep_when_free = 0.05 if platform.system() == "Windows" else 0.001
        self.progress_update_sleep_when_free = (
            0.1 if platform.system() == "Windows" else 0.01
        )
        self.max_size = max_size
        self.blocks = blocks
        self._asyncio_tasks: set[asyncio.Task] = set()
        # External durable queue for redelivery. None keeps the pure in-process
        # behavior; U12 wires it from configuration.
        self.job_queue = None
        # Idempotency keys whose work has been applied, and the lease-renewal
        # interval for a running durable job.
        self._applied_job_keys: set[str] = set()
        # Keys this process published and is running locally. The durable
        # consumer skips these, so a job is not executed twice on the replica
        # that queued it (see run_durable_job).
        self._local_job_keys: set[str] = set()
        self._local_job_messages: dict[str, str] = {}
        self._local_job_renewers: dict[str, asyncio.Task] = {}
        self.job_lease_renew_interval = 20.0
        self._durable_task: asyncio.Task | None = None
        self.default_concurrency_limit = self._resolve_concurrency_limit(
            default_concurrency_limit
        )
        self.event_analytics: dict[str, dict[str, float | str | None]] = {}
        self.ANALYTICS_MAX_EVENTS = max(
            1, int(os.getenv("GRADIO_ANALYTICS_MAX_EVENTS", "10000"))
        )
        self.events_recorded_per_fn: defaultdict[str | None, int] = defaultdict(int)
        self.cached_event_analytics_summary = {"functions": {}}
        self.events_recorded = 0
        self.event_count_at_last_cache = 0
        self.ANAYLTICS_CACHE_FREQUENCY = int(
            os.getenv("GRADIO_ANALYTICS_CACHE_FREQUENCY", "1")
        )

    @staticmethod
    def _get_df(records):
        import pandas as pd

        try:
            with pd.option_context("future.no_silent_downcasting", True):
                return (
                    pd.DataFrame(records).fillna(value=np.nan).infer_objects(copy=False)  # type: ignore
                )
        except Exception as e:
            if "No such keys(s)" in str(e):
                return (
                    pd.DataFrame(records).fillna(value=np.nan).infer_objects(copy=False)  # type: ignore
                )
            raise e

    def compute_analytics_summary(self, records):
        if not records:
            return self.cached_event_analytics_summary
        if (
            self.events_recorded - self.event_count_at_last_cache
            >= self.ANAYLTICS_CACHE_FREQUENCY
        ):
            df = self._get_df(records)
            self.event_count_at_last_cache = self.events_recorded
            grouped = df.groupby("function")
            metrics = {"functions": {}}
            for fn_name, fn_df in grouped:
                status = fn_df["status"].values
                success = np.sum(status == "success")
                failure = np.sum(status == "failed")
                total = success + failure
                success_rate = success / total if total > 0 else None
                percentiles = np.percentile(fn_df["process_time"].values, [50, 90, 99])  # type: ignore
                metrics["functions"][fn_name] = {
                    "success_rate": success_rate,
                    "process_time_percentiles": {
                        "50th": percentiles[0],  # type: ignore
                        "90th": percentiles[1],  # type: ignore
                        "99th": percentiles[2],  # type: ignore
                    },
                    "total_requests": self.events_recorded_per_fn.get(
                        fn_name, fn_df.shape[0]
                    ),
                }
            self.cached_event_analytics_summary = metrics
        return self.cached_event_analytics_summary

    def start(self):
        self.active_jobs = [None] * self.max_thread_count

        run_coro_in_background(self.start_processing)
        run_coro_in_background(self.start_progress_updates)
        if not self.live_updates:
            run_coro_in_background(self.notify_clients)

    def create_event_queue_for_fn(self, block_fn: BlockFunction):
        concurrency_id = block_fn.concurrency_id
        concurrency_limit: int | None
        if block_fn.concurrency_limit == "default":
            concurrency_limit = self.default_concurrency_limit
        else:
            concurrency_limit = block_fn.concurrency_limit
        if concurrency_id not in self.event_queue_per_concurrency_id:
            self.event_queue_per_concurrency_id[concurrency_id] = EventQueue(
                concurrency_id, concurrency_limit
            )
        elif (
            concurrency_limit is not None
        ):  # Update concurrency limit if it is lower than existing limit
            existing_event_queue = self.event_queue_per_concurrency_id[concurrency_id]
            if (
                existing_event_queue.concurrency_limit is None
                or concurrency_limit < existing_event_queue.concurrency_limit
            ):
                existing_event_queue.concurrency_limit = concurrency_limit

    def close(self, drain: bool = False, timeout: float | None = None):
        """Stop accepting new work.

        With ``drain=True`` the in-flight jobs are given up to ``timeout``
        seconds to finish before they are cancelled; the default stops as
        before.
        """
        self.stopped = True
        if drain:
            self.drain_timeout = timeout

    def send_message(
        self,
        event: Event,
        event_message: EventMessage,
    ):
        if not event.alive:
            return
        event_message.event_id = event._id
        if event.own_stream:
            # Once the event's request has closed there is no one to deliver to.
            own_messages = self.pending_messages_per_event.get(event._id)
            if own_messages is not None:
                own_messages.put_nowait(event_message)
            return
        messages = self.pending_messages_per_session[event.session_hash]
        messages.put_nowait(event_message)

    def _resolve_concurrency_limit(
        self, default_concurrency_limit: int | None | Literal["not_set"]
    ) -> int | None:
        """
        Handles the logic of resolving the default_concurrency_limit as this can be specified via a combination
        of the `default_concurrency_limit` parameter of the `Blocks.queue()` or the `GRADIO_DEFAULT_CONCURRENCY_LIMIT`
        environment variable. The parameter in `Blocks.queue()` takes precedence over the environment variable.
        Parameters:
            default_concurrency_limit: The default concurrency limit, as specified by a user in `Blocks.queue()`.
        """
        if default_concurrency_limit != "not_set":
            return default_concurrency_limit
        if default_concurrency_limit_env := os.environ.get(
            "GRADIO_DEFAULT_CONCURRENCY_LIMIT"
        ):
            if default_concurrency_limit_env.lower() == "none":
                return None
            else:
                return int(default_concurrency_limit_env)
        else:
            return 1

    def __len__(self):
        total_len = 0
        for event_queue in self.event_queue_per_concurrency_id.values():
            total_len += len(event_queue.queue)
        return total_len

    async def push(
        self,
        body: PredictBodyInternal,
        request: fastapi.Request,
        username: str | None,
        own_stream: bool = False,
    ) -> tuple[
        bool,
        str | list[dict[str, Any]],
        Literal["success", "error", "queue_full", "validator_error"],
    ]:
        if body.fn_index is None:
            return False, "No function index provided.", "error"
        if self.max_size is not None and len(self) >= self.max_size:
            return (
                False,
                f"Queue is full. Max size is {self.max_size} and size is {len(self)}.",
                "queue_full",
            )

        fn = route_utils.get_fn(self.blocks, None, body, principal=username)
        self.create_event_queue_for_fn(fn)
        if fn.validator is not None:
            gr_request = route_utils.compile_gr_request(
                body=body,
                fn=fn,
                username=username,
                request=None,
            )
            assert body.request is not None  # noqa: S101
            api_route_path = route_utils.get_api_call_path(request=body.request)
            root_path = route_utils.get_root_url(
                request=body.request,
                route_path=api_route_path,
                root_path=self.blocks.app.root_path,
            )
            validator_fn = create_validator_fn(fn)
            try:
                response = await route_utils.call_process_api(
                    app=self.blocks.app,
                    body=body,
                    gr_request=gr_request,
                    fn=validator_fn,
                    root_path=root_path,
                )

                validation_response: list[dict[str, Any]] | dict[str, Any] | None = (
                    response.get("data")
                )

                if validation_response is not None:
                    (is_valid, validation_data) = process_validation_response(
                        validation_response, fn
                    )
                    if is_valid is False:
                        return (
                            False,
                            validation_data,
                            "validator_error",
                        )

            except Exception as e:
                print(str(e))
                return False, str(e), "error"
        event = Event(
            body.session_hash,
            fn,
            request,
            username,
        )
        event.data = body
        if body.session_hash is None:
            body.session_hash = event.session_hash
        if own_stream:
            event.own_stream = True
            self.pending_messages_per_event[event._id] = AsyncQueue()
        async with self.pending_message_lock:
            if (
                not own_stream
                and body.session_hash not in self.pending_messages_per_session
            ):
                self.pending_messages_per_session[body.session_hash] = AsyncQueue()
            if body.session_hash not in self.pending_event_ids_session:
                self.pending_event_ids_session[body.session_hash] = set()
        self.pending_event_ids_session[body.session_hash].add(event._id)
        self.event_ids_to_events[event._id] = event
        body.event_id = event._id if not fn.batch else None

        if hasattr(fn.fn, "cache"):
            try:
                cache_start = time.time()
                gr_request = route_utils.compile_gr_request(
                    body=body,
                    fn=fn,
                    username=username,
                    request=None,
                )
                assert body.request is not None  # noqa: S101
                api_route_path = route_utils.get_api_call_path(request=body.request)
                root_path = route_utils.get_root_url(
                    request=body.request,
                    route_path=api_route_path,
                    root_path=self.blocks.app.root_path,
                )
                with ProbeCache():
                    response = await route_utils.call_process_api(
                        app=self.blocks.app,
                        body=body,
                        gr_request=gr_request,
                        fn=fn,
                        root_path=root_path,
                    )
                    while response and response.get("is_generating", False):
                        self.send_message(
                            event,
                            ProcessGeneratingMessage(
                                output=response,
                                success=True,
                            ),
                        )
                        response = await route_utils.call_process_api(
                            app=self.blocks.app,
                            body=body,
                            gr_request=gr_request,
                            fn=fn,
                            root_path=root_path,
                        )
                cache_duration = time.time() - cache_start
                avg_time = (
                    self.process_time_per_fn[fn].avg_time
                    if fn in self.process_time_per_fn
                    else None
                )
                self.send_message(
                    event,
                    ProcessCompletedMessage(
                        output=response,
                        success=True,
                        used_cache="full",
                        cache_duration=cache_duration,
                        avg_time=avg_time,
                    ),
                )
                return True, event._id, "success"
            except CacheMissError:
                pass  # Fall through to normal queue path
            except Exception:
                self.pending_messages_per_event.pop(event._id, None)
                raise

        try:
            event_queue = self.event_queue_per_concurrency_id[event.concurrency_id]
        except KeyError as e:
            self.pending_messages_per_event.pop(event._id, None)
            raise KeyError(
                "Event not found in queue. If you are deploying this Gradio app with multiple replicas without session affinity, enable multi-replica mode (see the Docker and Modal deployment guides); otherwise enable stickiness so all requests from the same user reach the same instance."
            ) from e
        event_queue.queue.append(event)
        self.event_analytics[event._id] = {
            "time": time.time(),
            "status": "queued",
            "process_time": None,
            "function": fn.api_name,
            "session_hash": body.session_hash,
        }
        self.events_recorded += 1
        self.events_recorded_per_fn[fn.api_name] += 1
        while len(self.event_analytics) > self.ANALYTICS_MAX_EVENTS:
            self.event_analytics.pop(next(iter(self.event_analytics)))

        # Durable path: also publish the job to the external queue so a job the
        # replica does not finish can be redelivered to a survivor. The local
        # copy still runs it; the idempotency key ties the two together.
        self.publish_durable_job(event, fn, body, username)
        self.broadcast_estimations(event.concurrency_id, len(event_queue.queue) - 1)
        return True, event._id, "success"

    def durable_jobs(self):
        """The configured external job queue, or None for the in-process path."""
        if self.job_queue is None:
            return None
        return self.job_queue

    def publish_durable_job(self, event, fn, body, username) -> None:
        """Publish a queued job for redelivery; no-op without an external queue.

        The job's idempotency key is derived from the event so a redelivered
        duplicate can be detected, and it is bound to the caller's principal so
        a survivor resolves the session as the same owner.
        """
        from gradio.queue_store import JobEnvelope

        queue = self.durable_jobs()
        if queue is None:
            return
        job = JobEnvelope(
            fn_index=fn._id,
            inputs=body.data,
            session_hash=body.session_hash,
            principal=username,
            batch=body.batched,
            event_id=event._id,
            idempotency_key=event._id,
        )
        # Mark the key before publishing so the consumer can never claim and run
        # this replica's own copy.
        self._local_job_keys.add(job.idempotency_key)
        try:
            message_id = queue.publish(job)
            self._local_job_messages[job.idempotency_key] = message_id
            self._local_job_renewers[job.idempotency_key] = run_coro_in_background(
                self._renew_lease, queue, message_id
            )
        except Exception:
            # The local copy still runs; a publish failure must not drop the
            # request that is already on the in-process queue.
            self._local_job_keys.discard(job.idempotency_key)
            self._local_job_messages.pop(job.idempotency_key, None)
            logger.exception("durable job publish failed")

    def start_durable_consumer(self) -> None:
        """Start the loop that runs jobs redelivered to this replica.

        The consumer reads new jobs and reclaims jobs an earlier consumer left
        pending past the lease, then runs each on this replica. No-op without an
        external queue.
        """
        if self.job_queue is None or (
            self._durable_task is not None and not self._durable_task.done()
        ):
            return
        self._durable_task = run_coro_in_background(self.consume_durable_jobs)
        self._asyncio_tasks.add(self._durable_task)
        self._durable_task.add_done_callback(self._durable_task_done)

    def _durable_task_done(self, task: asyncio.Task) -> None:
        self._asyncio_tasks.discard(task)
        if self._durable_task is task:
            self._durable_task = None

    async def consume_durable_jobs(self) -> None:
        queue = self.job_queue
        if queue is None:
            return
        lease_ms = getattr(queue, "lease_ms", 60_000)
        while not self.stopped:
            try:
                # New jobs run in the local queue so the submitting client can
                # receive its stream. This worker only takes over jobs whose
                # producer lease expired. Redis I/O stays off the event loop.
                messages = await asyncio.to_thread(queue.reclaim, min_idle_ms=lease_ms)
                for message in messages:
                    await self.run_durable_job(message)
                if not messages:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("durable consumer loop error")
                await asyncio.sleep(1.0)

    async def run_durable_job(self, message) -> bool:
        """Run one claimed job, then ack it only after the work is applied.

        The queue's message id is the idempotency key: a redelivered job whose
        work was already applied is acked without re-running, and the ack is
        ordered after `process_events` returns (which is after the session save
        inside `call_process_api`), so a crash between the two redelivers rather
        than losing state.
        """
        queue = self.job_queue
        if queue is None:
            return False
        job = message.job
        key = job.idempotency_key
        if key in self._applied_job_keys:
            queue.ack(message.id)
            return False
        if key in self._local_job_keys:
            # This replica queued and is running the job locally. Leave the
            # entry pending: if the local run never finishes, the lease expires
            # and a survivor reclaims it; if it finishes, the key moves to
            # _applied_job_keys and a later claim acks it.
            return False

        blocks = self.blocks
        fn = blocks.fns.get(job.fn_index)
        if fn is None:
            # The function is not in this app's config; leave it for a replica
            # that has it.
            return False

        self.create_event_queue_for_fn(fn)
        # A redelivered job has no client request; synthesize one from the
        # app's own local URL so processing has a request to compile against.
        request = _starlette_request_from_local_url(blocks)
        event = Event(
            job.session_hash,
            fn,
            request=request,  # type: ignore[arg-type]
            username=job.principal,
        )
        body = PredictBodyInternal(
            data=job.inputs,
            fn_index=job.fn_index,
            session_hash=job.session_hash,
            batched=bool(job.batch),
            request=request,  # type: ignore[arg-type]
        )
        body.event_id = event._id
        event.data = body
        self.event_ids_to_events[event._id] = event
        # A redelivered job has no live client channel; buffered messages are
        # dropped when the run finishes.
        self.pending_messages_per_session.setdefault(event.session_hash, AsyncQueue())

        renewer = asyncio.create_task(self._renew_lease(queue, message.id))
        try:
            await self.process_events([event], bool(job.batch), time.time(), fn)
        finally:
            renewer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewer
            self.event_ids_to_events.pop(event._id, None)

        # Work is applied (or terminally failed); record the key so a
        # redelivery does not re-run it, then ack.
        self._applied_job_keys.add(key)
        queue.ack(message.id)
        return True

    async def _renew_lease(self, queue, message_id: str) -> None:
        """Keep a running job's lease alive so a healthy replica is not preempted."""
        while True:
            await asyncio.sleep(self.job_lease_renew_interval)
            try:
                queue.renew(message_id)
            except Exception:
                logger.debug("durable job lease renew failed", exc_info=True)

    async def remove_from_queue(self, event_id: str):
        event = self.event_ids_to_events.get(event_id)
        if event:
            async with self.delete_lock:
                q = self.event_queue_per_concurrency_id[event.concurrency_id]
                try:
                    q.queue.remove(event)
                    self.event_ids_to_events.pop(event_id, None)
                except ValueError:
                    pass

    def _cancel_asyncio_tasks(self):
        for task in list(self._asyncio_tasks):
            task.cancel()
        self._asyncio_tasks.clear()

    async def drain(self) -> None:
        """Let in-flight jobs finish within the drain window, then cancel.

        With no window configured (the default) this cancels immediately, so
        the single-process shutdown path is unchanged.
        """
        tasks = set(self._asyncio_tasks)
        if self.drain_timeout is not None and tasks:
            _, pending = await asyncio.wait(tasks, timeout=self.drain_timeout)
        else:
            pending = tasks
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._asyncio_tasks.clear()

    def set_server_app(self, app: routes.App):
        self.server_app = app

    def get_active_worker_count(self) -> int:
        count = 0
        for worker in self.active_jobs:
            if worker is not None:
                count += 1
        return count

    def get_events(self) -> tuple[list[Event], bool, str] | None:
        concurrency_ids = list(self.event_queue_per_concurrency_id.keys())
        random.shuffle(concurrency_ids)
        for concurrency_id in concurrency_ids:
            event_queue = self.event_queue_per_concurrency_id[concurrency_id]
            if len(event_queue.queue) and (
                event_queue.concurrency_limit is None
                or event_queue.current_concurrency < event_queue.concurrency_limit
            ):
                first_event = event_queue.queue[0]
                block_fn = first_event.fn
                events = [first_event]
                batch = block_fn.batch
                if batch:
                    events += [
                        event
                        for event in event_queue.queue[1:]
                        if event.fn == first_event.fn
                    ][: block_fn.max_batch_size - 1]

                for event in events:
                    event_queue.queue.remove(event)

                return events, batch, concurrency_id

    async def start_processing(self) -> None:
        try:
            while not self.stopped:
                if len(self) == 0:
                    await asyncio.sleep(self.sleep_when_free)
                    continue

                if None not in self.active_jobs:
                    await asyncio.sleep(self.sleep_when_free)
                    continue

                # Using mutex to avoid editing a list in use
                async with self.delete_lock:
                    event_batch = self.get_events()

                if event_batch:
                    events, batch, concurrency_id = event_batch
                    self.active_jobs[self.active_jobs.index(None)] = events
                    event_queue = self.event_queue_per_concurrency_id[concurrency_id]
                    event_queue.current_concurrency += 1
                    start_time = time.time()
                    fn = events[0].fn
                    event_queue.start_times_per_fn[fn].add(start_time)
                    for event in events:
                        if (a := self.event_analytics.get(event._id)) is not None:
                            a["status"] = "processing"
                    process_event_task = run_coro_in_background(
                        self.process_events, events, batch, start_time, fn
                    )
                    set_task_name(
                        process_event_task,
                        events[0].session_hash,
                        fn._id,
                        events[0]._id,
                        batch,
                    )

                    self._asyncio_tasks.add(process_event_task)
                    process_event_task.add_done_callback(self._asyncio_tasks.discard)
                    if self.live_updates:
                        self.broadcast_estimations(concurrency_id)
                else:
                    await asyncio.sleep(self.sleep_when_free)
        finally:
            self.stopped = True
            await self.drain()

    async def start_progress_updates(self) -> None:
        """
        Because progress updates can be very frequent, we do not necessarily want to send a message per update.
        Rather, we check for progress updates at regular intervals, and send a message if there is a pending update.
        Consecutive progress updates between sends will overwrite each other so only the most recent update will be sent.
        """
        while not self.stopped:
            events = [evt for job in self.active_jobs if job is not None for evt in job]

            if len(events) == 0:
                await asyncio.sleep(self.progress_update_sleep_when_free)
                continue

            for event in events:
                if event.progress_pending and event.progress:
                    event.progress_pending = False
                    self.send_message(event, event.progress)

            await asyncio.sleep(self.progress_update_sleep_when_free)

    def set_progress(
        self,
        event_id: str,
        iterables: list[TrackedIterable] | None,
    ):
        if iterables is None:
            return
        for job in self.active_jobs:
            if job is None:
                continue
            for evt in job:
                if evt._id == event_id:
                    progress_data: list[ProgressUnit] = []
                    for iterable in iterables:
                        progress_unit = ProgressUnit(
                            index=iterable.index,
                            length=iterable.length,
                            unit=iterable.unit,
                            progress=iterable.progress,
                            desc=iterable.desc,
                        )
                        progress_data.append(progress_unit)
                    evt.progress = ProgressMessage(progress_data=progress_data)
                    evt.progress_pending = True

    def log_message(
        self,
        event_id: str,
        log: str,
        title: str,
        level: Literal["info", "warning", "success"],
        duration: float | None = 10,
        visible: bool = True,
    ):
        events = [evt for job in self.active_jobs if job is not None for evt in job]
        for event in events:
            if event._id == event_id:
                log_message = LogMessage(
                    log=log,
                    level=level,
                    duration=duration,
                    visible=visible,
                    title=title,
                )
                self.send_message(event, log_message)

    async def clean_events(
        self, *, session_hash: str | None = None, event_id: str | None = None
    ) -> None:
        for job_set in self.active_jobs:
            if job_set:
                for job in job_set:
                    if job.session_hash == session_hash or job._id == event_id:
                        job.alive = False

        async with self.delete_lock:
            events_to_remove: list[Event] = []
            for event_queue in self.event_queue_per_concurrency_id.values():
                for event in event_queue.queue:
                    if event.session_hash == session_hash or event._id == event_id:
                        events_to_remove.append(event)

            for event in events_to_remove:
                self.event_queue_per_concurrency_id[event.concurrency_id].queue.remove(
                    event
                )
                self.event_ids_to_events.pop(event._id, None)

            if session_hash and session_hash in self.pending_event_ids_session:
                removed_ids = {e._id for e in events_to_remove}
                self.pending_event_ids_session[session_hash] -= removed_ids
                if not self.pending_event_ids_session[session_hash]:
                    self.pending_event_ids_session.pop(session_hash, None)

    async def notify_clients(self) -> None:
        """
        Notify clients about events statuses in the queue periodically.
        """
        while not self.stopped:
            await asyncio.sleep(self.update_intervals)
            if len(self) > 0:
                for concurrency_id in self.event_queue_per_concurrency_id:
                    self.broadcast_estimations(concurrency_id)

    def broadcast_estimations(
        self, concurrency_id: str, after: int | None = None
    ) -> None:
        wait_so_far = 0
        event_queue = self.event_queue_per_concurrency_id[concurrency_id]
        time_till_available_worker: int | None = 0

        if event_queue.current_concurrency == event_queue.concurrency_limit:
            expected_end_times = []
            for fn, start_times in event_queue.start_times_per_fn.items():
                if fn not in self.process_time_per_fn:
                    time_till_available_worker = None
                    break
                if fn.connection == "stream":
                    process_time = fn.time_limit or 0
                else:
                    process_time = self.process_time_per_fn[fn].avg_time
                expected_end_times += [
                    start_time + process_time for start_time in start_times
                ]
            if time_till_available_worker is not None and len(expected_end_times) > 0:
                time_of_first_completion = min(expected_end_times)
                time_till_available_worker = max(
                    time_of_first_completion - time.time(), 0
                )

        for rank, event in enumerate(event_queue.queue):
            process_time_for_fn = (
                self.process_time_per_fn[event.fn].avg_time
                if event.fn in self.process_time_per_fn
                else None
            )

            # eta is the time remaining from now until the result will be returned
            # process_time_for_fn = time to run fn once worker assigned to it
            # wait_so_far = time till event gets to the head of the queue
            # time_till_available_worker = time for a worker to be assigned to it once its at the head
            # For streaming events, we modify this calculation slightly to be the time until the first
            # chunk is processed.
            rank_eta = (
                process_time_for_fn + wait_so_far + time_till_available_worker
                if process_time_for_fn is not None
                and wait_so_far is not None
                and time_till_available_worker is not None
                else None
            )

            if after is None or rank >= after:
                self.send_message(
                    event,
                    EstimationMessage(
                        rank=rank, rank_eta=rank_eta, queue_size=len(event_queue.queue)
                    ),
                )
            if event_queue.concurrency_limit is None:
                wait_so_far = 0
            elif wait_so_far is not None and process_time_for_fn is not None:
                delta = process_time_for_fn / event_queue.concurrency_limit
                if event.streaming:
                    delta = (
                        time_till_available_worker or 0
                    ) / event_queue.concurrency_limit
                wait_so_far += delta
            else:
                wait_so_far = None

    def get_status(self) -> EstimationMessage:
        return EstimationMessage(
            queue_size=len(self),
        )

    @staticmethod
    async def wait_for_event(event: Event) -> str:
        await event.signal.wait()
        return "signal"

    @staticmethod
    async def timeout(timeout: float) -> str:
        await asyncio.sleep(timeout)
        return "timeout"

    @staticmethod
    async def wait_for_event_or_timeout(
        event: Event, timeout: float
    ) -> Literal["signal", "timeout"]:
        t1 = asyncio.create_task(Queue.wait_for_event(event))
        t2 = asyncio.create_task(Queue.timeout(timeout))
        done, _ = await asyncio.wait(
            [t1, t2],
            return_when=asyncio.FIRST_COMPLETED,
        )
        done = [d.result() for d in done]
        event.signal.clear()
        return cast(Literal["signal", "timeout"], done[0])

    @staticmethod
    async def wait_for_batch(
        events: list[Event], timeouts: list[float]
    ) -> tuple[list[Event], list[Event]]:
        tasks = []
        for event, timeout in zip(events, timeouts, strict=False):
            tasks.append(
                asyncio.create_task(Queue.wait_for_event_or_timeout(event, timeout))
            )
        done, _ = await asyncio.wait(
            tasks,
            return_when=asyncio.ALL_COMPLETED,
        )
        done = [d.result() for d in done]
        awake_events = []
        closed_events = []
        for result, event in zip(done, events, strict=False):
            if result == "signal":
                awake_events.append(event)
            else:
                closed_events.append(event)
        return awake_events, closed_events

    async def process_events(
        self,
        events: list[Event],
        batch: bool,
        begin_time: float,
        fn: BlockFunction,
    ) -> None:
        awake_events: list[Event] = []
        success = False
        try:
            for event in events:
                if event.alive:
                    self.send_message(
                        event,
                        ProcessStartsMessage(
                            eta=self.process_time_per_fn[fn].avg_time
                            if fn in self.process_time_per_fn
                            else None
                        ),
                    )
                    awake_events.append(event)
            if not awake_events:
                return

            events = awake_events
            body = events[0].data
            if body is None:
                raise ValueError("No event data")
            username = events[0].username
            body.event_id = events[0]._id if not batch else None
            try:
                body.request = events[0].request
            except ValueError:
                pass

            if batch:
                body.data = list(
                    zip(
                        *[event.data.data for event in events if event.data],
                        strict=False,
                    )
                )
                body.request = events[0].request
                body.batched = True

            app = self.server_app
            if app is None:
                raise Exception("Server app has not been set.")

            gr_request = route_utils.compile_gr_request(
                body=body,
                fn=fn,
                username=username,
                request=None,
            )
            assert body.request is not None  # noqa: S101
            api_route_path = route_utils.get_api_call_path(request=body.request)
            root_path = route_utils.get_root_url(
                request=body.request,
                route_path=api_route_path,
                root_path=app.root_path,
            )
            first_iteration = 0

            if PROFILING_ENABLED:
                trace = RequestTrace(
                    event_id=events[0]._id,
                    fn_name=fn.api_name or str(fn.fn),
                    session_hash=events[0].session_hash,
                )
                trace.queue_wait_ms = (time.monotonic() - events[0].enqueue_time) * 1000
                set_current_trace(trace)
            else:
                trace = None

            try:
                start = time.monotonic()
                response = await route_utils.call_process_api(
                    app=app,
                    body=body,
                    gr_request=gr_request,
                    fn=fn,
                    root_path=root_path,
                )
                end = time.monotonic()
                first_iteration = end - start
                err = None
                for event in awake_events:
                    event.run_time += end - start
                    if event.streaming:
                        response["is_generating"] = not event.is_finished

            except Exception as e:
                if not isinstance(e, Error) or e.print_exception:
                    traceback.print_exc()
                response = None
                err = e
                for event in awake_events:
                    content = error_payload(err, app.get_blocks().show_error)
                    self.send_message(
                        event,
                        ProcessCompletedMessage(
                            output=content,
                            title=content.get("title", "Error"),  # type: ignore
                            success=False,
                        ),
                    )
                    await run_sync(
                        self.compute_analytics_summary,
                        list(self.event_analytics.values()),
                    )
            if response and response.get("is_generating", False):
                old_response = response
                old_err = err
                while response and response.get("is_generating", False):
                    start = time.monotonic()
                    old_response = response
                    old_err = err
                    for event in awake_events:
                        self.send_message(
                            event,
                            ProcessGeneratingMessage(
                                msg=ServerMessage.process_generating
                                if not event.streaming
                                else ServerMessage.process_streaming,
                                output=old_response,
                                success=old_response is not None,
                                time_limit=None
                                if not fn.time_limit
                                else cast(int, fn.time_limit) - first_iteration
                                if event.streaming
                                else None,
                            ),
                        )
                    awake_events = [event for event in awake_events if event.alive]
                    if not awake_events:
                        return
                    try:
                        start = time.monotonic()
                        if awake_events[0].streaming:
                            awake_events, closed_events = await Queue.wait_for_batch(
                                awake_events,
                                # We need to wait for all of the events to have the latest input data
                                # the max time is the time limit of the function or 30 seconds (arbitrary) but should
                                # never really take that long to make a request from the client to the server unless
                                # the client disconnected.
                                [cast(float, fn.time_limit or 30) - first_iteration]
                                * len(awake_events),
                            )
                            for closed_event in closed_events:
                                self.send_message(
                                    closed_event,
                                    ProcessCompletedMessage(
                                        output=response, success=True
                                    ),
                                )
                        if not awake_events:
                            break
                        # Re-read the event's fn, which may have been swapped out
                        # if the app was hot-reloaded while this event was generating
                        fn = awake_events[0].fn
                        body = cast(PredictBodyInternal, awake_events[0].data)
                        if batch:
                            body.data = list(
                                zip(
                                    *[
                                        event.data.data
                                        for event in events
                                        if event.data
                                    ],
                                    strict=False,
                                )
                            )
                        response = await route_utils.call_process_api(
                            app=app,
                            body=body,
                            gr_request=gr_request,
                            fn=fn,
                            root_path=root_path,
                        )
                        end = time.monotonic()
                        for event in awake_events:
                            event.run_time += end - start
                            if event.streaming:
                                response["is_generating"] = not event.is_finished
                    except Exception as e:
                        if not isinstance(e, Error) or e.print_exception:
                            traceback.print_exc()
                        response = None
                        err = e

                if response:
                    success = True
                    output = response
                else:
                    success = False
                    error = err or old_err
                    output = error_payload(error, app.get_blocks().show_error)
                cache_source = output
                if (
                    success
                    and not output.get("used_cache")
                    and old_response
                    and old_response.get("used_cache")
                ):
                    cache_source = old_response
                used_cache = cache_source.get("used_cache") if success else None
                used_cache = (
                    cast(Literal["full", "partial"], used_cache)
                    if used_cache in ("full", "partial")
                    else None
                )
                for event in awake_events:
                    self.send_message(
                        event,
                        ProcessCompletedMessage(
                            output=output,
                            success=success,
                            used_cache=used_cache,
                            cache_duration=cache_source.get("duration"),  # type: ignore[arg-type]
                            avg_time=cache_source.get("average_duration"),  # type: ignore[arg-type]
                        ),
                    )

            elif response:
                for e, event in enumerate(awake_events):
                    # Copy per event because "data" is replaced below and the
                    # message is serialized later; only that key changes.
                    output = dict(response)
                    if batch and "data" in output:
                        output["data"] = list(zip(*response.get("data"), strict=False))[
                            e
                        ]
                    success = response is not None
                    used_cache = output.get("used_cache") if success else None
                    used_cache = (
                        cast(Literal["full", "partial"], used_cache)
                        if used_cache in ("full", "partial")
                        else None
                    )
                    self.send_message(
                        event,
                        ProcessCompletedMessage(
                            output=output,
                            success=success,
                            used_cache=used_cache,
                            cache_duration=output.get("duration"),  # type: ignore[arg-type]
                            avg_time=output.get("average_duration"),  # type: ignore[arg-type]
                        ),
                    )
            end_time = time.time()
            if response is not None:
                duration = (
                    end_time - begin_time
                    if not events[0].streaming
                    else first_iteration
                )
                if not response.get("used_cache"):
                    self.process_time_per_fn[events[0].fn].add(duration)
                for event in events:
                    if (a := self.event_analytics.get(event._id)) is not None:
                        a["process_time"] = duration
        except Exception as e:
            if not isinstance(e, Error) or e.print_exception:
                traceback.print_exc()
        finally:
            if PROFILING_ENABLED:
                trace = get_current_trace()
                if trace is not None:
                    collector.add(trace)

            event_queue = self.event_queue_per_concurrency_id[events[0].concurrency_id]
            event_queue.current_concurrency -= 1
            start_times = event_queue.start_times_per_fn.get(fn)
            if start_times is not None:
                start_times.discard(begin_time)
                if not start_times:
                    del event_queue.start_times_per_fn[fn]
            try:
                self.active_jobs[self.active_jobs.index(events)] = None
            except ValueError:
                # `events` can be absent from `self.active_jobs`
                # when this coroutine is called from the `join_queue` endpoint handler in `routes.py`
                # without putting the `events` into `self.active_jobs`.
                # https://github.com/gradio-app/gradio/blob/f09aea34d6bd18c1e2fef80c86ab2476a6d1dd83/gradio/routes.py#L594-L596
                pass
            app = self.server_app
            if app is not None:
                blocks = app.get_blocks()
                for event in events:
                    # A run that raised, was cancelled or lost its client reaches
                    # no final chunk, so close out its streams here, while the
                    # iterator that keys them is still stored (a finished run's
                    # is already None). /cancel awaits this task before dropping
                    # that iterator itself.
                    blocks._drop_run_streams(
                        event.session_hash, app.iterators.get(event._id)
                    )
            for event in events:
                # Always reset the state of the iterator
                # If the job finished successfully, this has no effect
                # If the job is cancelled, this will enable future runs
                # to start "from scratch"
                await self.reset_iterators(event._id)

                if (a := self.event_analytics.get(event._id)) is not None:
                    a["status"] = (
                        ("success" if success else "failed")
                        if event in awake_events
                        else "cancelled"
                    )
                await run_sync(
                    self.compute_analytics_summary,
                    list(self.event_analytics.values()),
                )

                message_id = self._local_job_messages.pop(event._id, None)
                if message_id is not None:
                    self._local_job_keys.discard(event._id)
                    renewer = self._local_job_renewers.pop(event._id, None)
                    if renewer is not None:
                        renewer.cancel()
                        await asyncio.gather(renewer, return_exceptions=True)
                    if success:
                        # process_events returns after call_process_api saves
                        # session state, so ack only after the durable write.
                        self._applied_job_keys.add(event._id)
                        try:
                            await asyncio.to_thread(self.job_queue.ack, message_id)
                        except Exception:
                            logger.exception("durable job ack failed after local run")

                self.event_ids_to_events.pop(event._id, None)

    async def reset_iterators(self, event_id: str):
        # Do the same thing as the /reset route
        app = self.server_app
        if app is None:
            raise Exception("Server app has not been set.")
        if event_id not in app.iterators:
            # Failure, but don't raise an error
            return
        async with app.lock:
            try:
                await safe_aclose_iterator(app.iterators[event_id])
            except Exception:
                pass
            del app.iterators[event_id]
        return


def _starlette_request_from_local_url(blocks: Blocks) -> fastapi.Request:
    """A minimal request for a job that has no client connection (redelivery).

    Uses the app's own local URL so route/root-path derivation works; there is
    no client to receive messages, which `run_durable_job` accounts for.
    """
    from starlette.requests import Request as StarletteRequest

    path = f"{route_utils.API_PREFIX}/queue/join"
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"localhost")],
        "server": ("localhost", 0),
        "client": ("localhost", 0),
    }

    async def receive() -> dict:
        return {"type": "http.request"}

    return StarletteRequest(scope, receive=receive)  # type: ignore[arg-type]


def create_validator_fn(fn: BlockFunction) -> BlockFunction:
    """
    Builds the BlockFunction that runs `fn.validator` before `fn` itself. It mirrors
    `fn`'s inputs exactly, including the `inputs_kwargs` mapping, so the validator is
    called with the same positional and keyword arguments as the main function.
    """
    if fn.validator is None:
        raise ValueError("Cannot build a validator function without a validator.")
    return BlockFunction(
        fn=fn.validator,
        api_name=None,
        api_visibility="undocumented",
        batch=fn.batch,
        concurrency_id=None,
        concurrency_limit=None,
        inputs=fn.inputs,
        outputs=fn.inputs,
        preprocess=fn.preprocess,
        postprocess=False,
        inputs_as_dict=fn.inputs_as_dict,
        input_keyword_names=fn.input_keyword_names,
        input_parameter_names=fn.input_parameter_names,
        targets=[],
        _id=-1,
        max_batch_size=fn.max_batch_size,
        tracks_progress=fn.tracks_progress,
        js=None,
        show_progress="hidden",
        show_progress_on=fn.show_progress_on,
        cancels=fn.cancels,
        collects_event_data=fn.collects_event_data,
        is_validator_function=True,
    )


def process_validation_response(
    validation_response: list[dict[str, Any]] | dict[str, Any],
    fn: BlockFunction | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    validation_data: list[dict[str, Any]] = []

    # Names are attached in `fn.inputs` order, because that is the order the frontend
    # walks when it paints each validation result onto `dep.inputs[i]`. With
    # `inputs_kwargs` that differs from the signature order of `fn.fn`.
    param_names: list[str | None] = []
    if fn:
        if fn.input_parameter_names:
            param_names = list(fn.input_parameter_names)
        elif fn.fn:
            param_names = list(inspect.signature(fn.fn).parameters)

    if isinstance(validation_response, list):
        for i, data in enumerate(validation_response):
            if isinstance(data, dict) and data.get("__type__", None) == "validate":
                param_name = param_names[i] if i < len(param_names) else None
                if param_name is None:
                    param_name = f"parameter_{i}"
                data_with_name = {**data, "parameter_name": param_name}
                validation_data.append(data_with_name)
            else:
                validation_data.append({"is_valid": True, "message": ""})

    elif (
        isinstance(validation_response, dict)
        and validation_response.get("is_valid", None) is False
    ):
        validation_data.append(
            validation_response,
        )
    else:
        validation_data.append({"is_valid": True, "message": ""})

    return all(
        x.get("is_valid", None) is True for x in validation_data
    ), validation_data
