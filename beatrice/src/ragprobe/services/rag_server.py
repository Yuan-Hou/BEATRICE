"""FastAPI server for the RAG probe scorer.

Endpoints:
  GET  /health
  POST /score        {question, contexts?, answer?, ...} -> answer + P(correct) from the
                     trained frozen-fusion probe (+ per-branch probabilities)
  POST /batch_score  {items: [...]} -> {items: [...]}

If `answer` is omitted the underlying LLM generates one (greedy, training-format prompt)
— but the intended workflow is: generate answers with vLLM, send them here for scoring.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable

from ragprobe.services.probe_scorer import ProbeScorer
from ragprobe.utils.logging import log


@dataclass
class _Job:
    item: dict[str, Any]
    future: Future


class _MicroBatcher:
    """Coalesces concurrent requests into batches handed to `run_batch`."""

    def __init__(self, name: str, run_batch: Callable[[list[dict]], list[dict]], config: dict[str, Any]):
        b = config.get("batch", {})
        self.name = name
        self.run_batch = run_batch
        self.max_batch_size = int(b.get("max_batch_size", 8))
        self.max_wait_s = float(b.get("max_wait_ms", 50)) / 1000.0
        self.queue_timeout_s = float(b.get("queue_timeout", 600))
        self.queue: queue.Queue[_Job | None] = queue.Queue(maxsize=int(b.get("max_queue_size", 4096)))
        self.total_requests = 0
        self.total_batches = 0
        self.last_batch_size = 0
        self.last_batch_latency_s = 0.0
        self.worker = threading.Thread(target=self._loop, name=f"probe-batcher-{name}", daemon=True)
        self.worker.start()

    def submit(self, item: dict[str, Any]) -> dict[str, Any]:
        fut: Future = Future()
        self.total_requests += 1
        self.queue.put(_Job(item, fut), timeout=self.queue_timeout_s)
        return fut.result(timeout=self.queue_timeout_s)

    def submit_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        futs = []
        for item in items:
            fut: Future = Future()
            self.total_requests += 1
            self.queue.put(_Job(item, fut), timeout=self.queue_timeout_s)
            futs.append(fut)
        return [f.result(timeout=self.queue_timeout_s) for f in futs]

    def close(self):
        self.queue.put(None)
        self.worker.join(timeout=5)

    def stats(self) -> dict[str, Any]:
        return {
            "queue_size": self.queue.qsize(),
            "max_batch_size": self.max_batch_size,
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "last_batch_size": self.last_batch_size,
            "last_batch_latency_s": round(self.last_batch_latency_s, 4),
        }

    def _loop(self):
        while True:
            job = self.queue.get()
            if job is None:
                return
            jobs = [job]
            deadline = time.monotonic() + self.max_wait_s
            while len(jobs) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self.queue.put(None)
                    break
                jobs.append(nxt)

            started = time.monotonic()
            try:
                results = self.run_batch([j.item for j in jobs])
            except Exception as exc:
                log(f"{self.name} batch crashed: {exc}")
                results = [{"status": "failed", "error": str(exc),
                            "example_id": j.item.get("example_id")} for j in jobs]
            self.total_batches += 1
            self.last_batch_size = len(jobs)
            self.last_batch_latency_s = time.monotonic() - started
            for j, r in zip(jobs, results):
                j.future.set_result(r)


def create_app(config: dict[str, Any]):
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Please install fastapi and uvicorn to run the probe scorer service") from exc

    scorer = ProbeScorer(config)
    batcher = _MicroBatcher("score", scorer.score_many, config)
    app = FastAPI(title="RAG Probe Scorer")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_name": scorer.model_name,
            "device": scorer.device,
            "hidden_layer": scorer.hidden_layer,
            "probes": {
                mode: {"checkpoint": s.checkpoint_path,
                       "branches": s.branch_names,
                       "binary_target": s.binary_target_name}
                for mode, s in scorer.stacks.items()
            },
            "batcher": batcher.stats(),
            "dual_fusion": (scorer.dual_head.ckpt_path if scorer.dual_head else None),
            "qonly_cache": scorer.qcache.stats(),
        }

    @app.post("/score")
    def score(payload: dict[str, Any]):
        return batcher.submit(payload)

    @app.post("/batch_score")
    def batch_score(payload: dict[str, Any]):
        return {"items": batcher.submit_many(payload.get("items") or [])}

    # ---- two-phase RAG flow: call /score_question once, then /score_context N times ----
    @app.post("/score_question")
    def score_question(payload: dict[str, Any]):
        """Phase 1: closed-book question score; caches q-only vecs, returns `qonly_uuid`."""
        return scorer.score_question(payload)

    @app.post("/score_context")
    def score_context(payload: dict[str, Any]):
        """Phase 2: question+context score. Pass `qonly_uuid` from phase 1 to fuse the
        cached q-only signal (dual q+ctx probe); may be called repeatedly per uuid."""
        return scorer.score_context(payload)

    @app.on_event("shutdown")
    def shutdown():
        batcher.close()

    return app
