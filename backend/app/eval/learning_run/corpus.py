"""Load immutable evaluation corpora into an isolated in-memory retriever."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import inspect
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any

import chromadb

from app.rag.runtime import build_retriever_for_collection

from .contracts import CorpusSnapshot
from .registry import RegistryError


_EMBEDDING_MODELS = {
    "ollama-nomic-embed-text-v1": "nomic-embed-text",
}


class CorpusMaterializerBusyError(RuntimeError):
    """The app-scoped materializer already owns its single worker."""


_MATERIALIZATION_LATE = object()


def _close_resource(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


class IsolatedCorpusRetriever:
    """Keep ephemeral resources alive and serialize search with cleanup."""

    def __init__(self, *, client: Any, collection: Any, retriever: Any) -> None:
        self._client = client
        self._collection = collection
        self.collection_name = collection.name
        self._retriever = retriever
        self._operation_lock = threading.RLock()
        self._closed = False

    def search(self, query: str, top_k: int = 5):
        with self._operation_lock:
            if self._closed or self._retriever is None:
                raise RuntimeError("isolated corpus retriever is closed")
            return self._retriever.search(query, top_k=top_k)

    def close(self) -> None:
        with self._operation_lock:
            if self._closed:
                return
            self._closed = True
            retriever = self._retriever
            client = self._client
            collection = self._collection
            self._retriever = None
            self._collection = None
            self._client = None
            _close_resource(retriever)
            if client is not None and collection is not None:
                try:
                    client.delete_collection(collection.name)
                except Exception:
                    pass
            _close_resource(client)


class CorpusMaterializerLease:
    """One active materialization owned by a controller."""

    def __init__(self, controller: "CorpusMaterializerController", retriever: Any) -> None:
        self._controller = controller
        self.retriever = retriever
        self._released = False
        self._release_lock = threading.Lock()

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._controller._release(self)


class CorpusMaterializerController:
    """Single-admission, draining-aware controller for isolated materialization."""

    def __init__(self, loader: Any) -> None:
        if loader is None or not callable(getattr(loader, "load", None)):
            raise TypeError("loader must provide load")
        self.loader = loader
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="learning-eval-corpus")
        self._lock = threading.RLock()
        self._state = "idle"
        self._outstanding = 0
        self._future: Future[Any] | None = None
        self._stop: threading.Event | None = None
        self._lease: CorpusMaterializerLease | None = None
        self._cleanup_scheduled = False
        self._cleanup_future: Future[Any] | None = None
        self._shutdown_requested = False
        self._executor_shutdown = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def outstanding_count(self) -> int:
        with self._lock:
            return self._outstanding

    @staticmethod
    def _invoke_loader(loader: Any, snapshot: CorpusSnapshot, stop: threading.Event, deadline: float):
        kwargs: dict[str, Any] = {"snapshot": snapshot}
        try:
            parameters = inspect.signature(loader.load).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "stop" in parameters or accepts_kwargs:
            kwargs["stop"] = stop
        if "deadline" in parameters or accepts_kwargs:
            kwargs["deadline"] = deadline
        return loader.load(**kwargs)

    def _materialize(
        self,
        snapshot: CorpusSnapshot,
        stop: threading.Event,
        deadline: float,
    ) -> Any:
        """Run one loader call and close a result that arrived too late.

        This wrapper is deliberately executed by the controller's sole worker.
        It is the only place that closes a result which was returned after a
        timeout/cancellation decision; future callbacks only hand off already
        materialized resources to another internal executor job.
        """

        result = self._invoke_loader(self.loader, snapshot, stop, deadline)
        if stop.is_set() or time.monotonic() >= float(deadline):
            _close_resource(result)
            return _MATERIALIZATION_LATE
        return result

    def _claim_executor_shutdown_locked(self) -> bool:
        if self._executor_shutdown:
            return False
        self._executor_shutdown = True
        return True

    def _finish_without_cleanup_locked(self) -> bool:
        """Transition a drained job with no owned resource to its terminal state."""

        self._state = "closed" if self._shutdown_requested else "idle"
        self._outstanding = 0
        self._future = None
        self._stop = None
        self._lease = None
        if self._shutdown_requested:
            return self._claim_executor_shutdown_locked()
        return False

    def _cleanup_worker(self, resource: Any) -> None:
        # Cleanup may wait for an in-flight search operation, but this function
        # always runs on the controller's dedicated worker, never the event loop.
        _close_resource(resource)
        should_shutdown = False
        with self._lock:
            if not self._cleanup_scheduled:
                return
            self._cleanup_scheduled = False
            self._cleanup_future = None
            self._state = "closed" if self._shutdown_requested else "idle"
            self._outstanding = 0
            self._future = None
            self._stop = None
            self._lease = None
            if self._shutdown_requested:
                should_shutdown = self._claim_executor_shutdown_locked()
        if should_shutdown:
            # Do not cancel the cleanup job that is currently running.
            self._executor.shutdown(wait=False, cancel_futures=False)

    def _schedule_cleanup_locked(self, resource: Any) -> bool:
        if self._cleanup_scheduled:
            return False
        self._cleanup_scheduled = True
        self._cleanup_future = self._executor.submit(self._cleanup_worker, resource)
        return True

    def _handle_draining_future_locked(self, future: Future[Any]) -> bool:
        """Bookkeep a completed loader future and schedule, never perform, cleanup."""

        try:
            result = future.result()
        except BaseException:
            return self._finish_without_cleanup_locked()
        if result is _MATERIALIZATION_LATE:
            return self._finish_without_cleanup_locked()
        self._schedule_cleanup_locked(result)
        return False

    def _done_callback(self, future: Future[Any]) -> None:
        should_shutdown = False
        with self._lock:
            # A cleanup completion may have cleared the materialization future;
            # stale callbacks must not mutate the next admission.
            if future is not self._future:
                return
            if self._state == "materializing":
                try:
                    error = future.exception()
                except BaseException:
                    # The exception has been consumed for bookkeeping.  The
                    # awaiting caller still receives the original exception.
                    error = RuntimeError("materialization future cancelled")
                if error is None:
                    return
                should_shutdown = self._finish_without_cleanup_locked()
            elif self._state == "draining":
                should_shutdown = self._handle_draining_future_locked(future)
        if should_shutdown:
            self._executor.shutdown(wait=False, cancel_futures=False)

    def _begin_draining(self) -> None:
        should_shutdown = False
        with self._lock:
            if self._state not in {"materializing", "active"}:
                return
            self._state = "draining"
            if self._stop is not None:
                self._stop.set()
            if self._lease is not None:
                self._schedule_cleanup_locked(self._lease.retriever)
            elif self._future is not None and self._future.done():
                should_shutdown = self._handle_draining_future_locked(self._future)
        if should_shutdown:
            self._executor.shutdown(wait=False, cancel_futures=False)

    async def acquire(self, *, snapshot: CorpusSnapshot, deadline: float) -> CorpusMaterializerLease:
        loop = asyncio.get_running_loop()
        remaining = float(deadline) - loop.time()
        with self._lock:
            if self._shutdown_requested or self._state != "idle":
                message = (
                    "isolated corpus materializer is closed"
                    if self._shutdown_requested
                    else "isolated corpus materializer is busy"
                )
                raise CorpusMaterializerBusyError(message)
            if remaining <= 0:
                raise TimeoutError("retrieval preflight deadline exceeded")
            stop = threading.Event()
            future = self._executor.submit(
                self._materialize,
                snapshot,
                stop,
                float(deadline),
            )
            self._state = "materializing"
            self._outstanding = 1
            self._future = future
            self._stop = stop
            future.add_done_callback(self._done_callback)
        wrapped = asyncio.wrap_future(future)

        def consume_wrapped_exception(done: asyncio.Future[Any]) -> None:
            if not done.cancelled():
                done.exception()

        # ``wait_for(shield(...))`` detaches its callback when the timeout
        # fires.  Consume a late wrapper exception independently so a stopped
        # loader cannot surface an unhandled asyncio Future warning.
        wrapped.add_done_callback(consume_wrapped_exception)
        try:
            retriever = await asyncio.wait_for(asyncio.shield(wrapped), timeout=remaining)
        except asyncio.TimeoutError:
            self._begin_draining()
            raise
        except asyncio.CancelledError:
            self._begin_draining()
            raise
        except Exception:
            with self._lock:
                if self._future is future and self._state == "materializing":
                    self._finish_without_cleanup_locked()
            raise
        if retriever is _MATERIALIZATION_LATE:
            self._begin_draining()
            raise TimeoutError("retrieval preflight deadline exceeded")
        if stop.is_set() or loop.time() >= float(deadline):
            self._begin_draining()
            raise TimeoutError("retrieval preflight deadline exceeded")
        should_shutdown = False
        with self._lock:
            if self._state != "materializing" or self._shutdown_requested:
                if self._future is future and future.done():
                    should_shutdown = self._handle_draining_future_locked(future)
                busy = True
            else:
                busy = False
                self._state = "active"
                lease = CorpusMaterializerLease(self, retriever)
                self._lease = lease
        if should_shutdown:
            self._executor.shutdown(wait=False, cancel_futures=False)
        if busy:
            raise CorpusMaterializerBusyError("isolated corpus materializer is draining")
        return lease

    async def load(self, *, snapshot: CorpusSnapshot, deadline: float) -> CorpusMaterializerLease:
        return await self.acquire(snapshot=snapshot, deadline=deadline)

    def _release(self, lease: CorpusMaterializerLease) -> None:
        with self._lock:
            if self._lease is not lease or self._state != "active":
                return
            self._state = "draining"
            if self._stop is not None:
                self._stop.set()
            self._schedule_cleanup_locked(lease.retriever)

    def shutdown(self, wait: bool = False) -> None:
        should_shutdown = False
        with self._lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            if self._stop is not None:
                self._stop.set()
            if self._state == "idle":
                self._state = "closed"
                self._outstanding = 0
                should_shutdown = self._claim_executor_shutdown_locked()
            elif self._state == "materializing":
                self._state = "draining"
                if self._future is not None and self._future.done():
                    should_shutdown = self._handle_draining_future_locked(self._future)
            elif self._state == "active":
                self._state = "draining"
                if self._lease is not None:
                    self._schedule_cleanup_locked(self._lease.retriever)
                elif self._future is not None and self._future.done():
                    should_shutdown = self._handle_draining_future_locked(self._future)
            elif self._state == "draining" and self._future is not None and self._future.done():
                should_shutdown = self._handle_draining_future_locked(self._future)
        if should_shutdown:
            # ``cancel_futures=False`` preserves an already submitted internal
            # cleanup and avoids leaking a RuntimeError to late callbacks.
            self._executor.shutdown(wait=wait, cancel_futures=False)


def _resolve_embedding_model(snapshot: CorpusSnapshot) -> str:
    try:
        expected_model = _EMBEDDING_MODELS[snapshot.embedding_config_version]
    except KeyError as exc:
        raise RegistryError(
            "unknown corpus embedding configuration: "
            f"{snapshot.embedding_config_version}"
        ) from exc
    configured_model = os.environ.get("EMBED_MODEL")
    if configured_model and configured_model != expected_model:
        raise RegistryError(
            "EMBED_MODEL conflicts with the corpus snapshot embedding configuration"
        )
    return expected_model


class CorpusSnapshotLoader:
    """Validate and materialize a snapshot without touching production RAG."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], Any] | None = None,
        builder: Callable[..., Any] | None = None,
    ) -> None:
        self.client_factory = client_factory or chromadb.EphemeralClient
        # ``None`` selects the production builder path.  A custom builder is
        # deliberately kept separate so its one-argument contract cannot be
        # hidden by exception-driven fallback.
        self.builder = builder

    @staticmethod
    def collection_name(snapshot: CorpusSnapshot) -> str:
        # Content-addressing by aggregate hash makes a snapshot collection
        # deterministic while keeping it distinct from production's fixed name.
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "-", snapshot.snapshot_id).strip("-_")
        safe_id = (safe_id or "snapshot")[:12]
        return f"learning_eval_{safe_id}_{snapshot.aggregate_hash[:32]}"

    @staticmethod
    def _chunk_payloads(snapshot: CorpusSnapshot) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "source": chunk.source,
                "page": chunk.page,
            }
            for chunk in snapshot.chunks
        ]

    def load(
        self,
        *,
        snapshot: CorpusSnapshot,
        global_retriever: Any | None = None,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> IsolatedCorpusRetriever:
        del global_retriever  # Compatibility sentinel: production retriever is forbidden.

        def check() -> None:
            if stop is not None and stop.is_set():
                raise TimeoutError("corpus materialization stopped")
            if deadline is not None and time.monotonic() >= float(deadline):
                raise TimeoutError("corpus materialization deadline exceeded")

        check()

        # All integrity checks happen before client creation or collection
        # creation.  A failed preflight therefore cannot mutate Chroma state.
        try:
            snapshot.validate_hashes()
        except (TypeError, ValueError) as exc:
            raise RegistryError(f"corpus snapshot validation failed: {exc}") from exc
        check()

        embedding_model = _resolve_embedding_model(snapshot)
        check()

        chunk_payloads = self._chunk_payloads(snapshot)
        chunk_ids = [chunk["chunk_id"] for chunk in chunk_payloads]
        if not chunk_ids or len(chunk_ids) != len(set(chunk_ids)):
            raise RegistryError("corpus snapshot contains missing or duplicate chunk IDs")

        client = None
        collection = None
        retriever = None
        isolated = None
        try:
            check()
            client = self.client_factory()
            check()
            collection_name = self.collection_name(snapshot)
            if collection_name == "study_coach_chunks":
                raise RegistryError("evaluation collection cannot be production collection")
            collection = client.get_or_create_collection(collection_name)
            check()
            if self.builder is None:
                retriever = build_retriever_for_collection(
                    collection,
                    embed_model=embedding_model,
                    embed_host=os.environ.get("OLLAMA_HOST") or None,
                )
            else:
                retriever = self.builder(collection)
            check()
            retriever.add_chunks(chunk_payloads)
            check()
            isolated = IsolatedCorpusRetriever(
                client=client,
                collection=collection,
                retriever=retriever,
            )
            return isolated
        except BaseException:
            if isolated is not None:
                isolated.close()
            else:
                _close_resource(retriever)
                if client is not None and collection is not None:
                    try:
                        client.delete_collection(collection.name)
                    except Exception:
                        pass
                _close_resource(client)
            raise
