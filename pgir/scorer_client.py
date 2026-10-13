"""HTTP client for the probe_scorer service (spec §3.3).

- 180s timeout, 3 retries with exponential backoff, failures logged to
  failures.jsonl and skipped (never abort a run on one bad item);
- asyncio concurrency semaphore (default 16) — the server micro-batches;
- pins the probe checkpoints seen at startup: any response scored by a
  different checkpoint raises (run metadata integrity, spec §11).

Two-phase flow (2026-07-10 service upgrade): items may carry
`"_endpoint": "score_question" | "score_context"` to hit the dual q+ctx
fusion endpoints (underscore keys are stripped from the payload).
These endpoints bypass the server's micro-batcher — concurrent requests
run concurrent eager forwards on the scorer GPU — so a process-wide
in-flight cap (PGIR_SCORER_INFLIGHT, default 3) bounds them client-side.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_TIMEOUT_S = 180.0
RETRIES = 3
BACKOFF_S = (1.0, 4.0, 16.0)
# max concurrent /score_question + /score_context requests across ALL threads.
# MUST stay 1 against the 2026-07-10 service: its two-phase endpoints bypass the
# micro-batcher and the hook-based feature extraction on the shared model is NOT
# thread-safe — concurrent requests contaminate each other's activation buffers
# ("size of tensor a (X) must match tensor b (Y)", A/B-verified 4/4 concurrent
# failures vs 0/4 serial). Raise only after the service serializes internally.
TWO_PHASE_INFLIGHT = int(os.environ.get("PGIR_SCORER_INFLIGHT", "1"))
_TWO_PHASE_SEM = threading.Semaphore(TWO_PHASE_INFLIGHT)


class ProbeCheckpointChanged(RuntimeError):
    pass


class ScorerClient:
    dual_fusion: str | None = None   # class default: robust to __new__-style construction

    def __init__(
        self,
        base_url: str = "http://localhost:8008",
        concurrency: int = 16,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        failures_path: str | Path | None = None,
        pin_checkpoints: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.concurrency = int(concurrency)
        self.timeout_s = float(timeout_s)
        self.failures_path = Path(failures_path) if failures_path else None
        self.pinned: dict[str, str] = {}
        self.dual_fusion = None
        self._health: dict[str, Any] = {}
        if pin_checkpoints:
            self._health = self.health()
            self.pinned = {
                mode: info["checkpoint"] for mode, info in self._health.get("probes", {}).items()
            }
            self.dual_fusion = self._health.get("dual_fusion")

    # ------------------------------------------------------------- sync API
    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as c:
            r = c.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    def run_metadata(self) -> dict[str, Any]:
        """What a run must record for reproducibility."""
        return {
            "scorer_url": self.base_url,
            "model_name": self._health.get("model_name"),
            "probe_checkpoints": dict(self.pinned),
            "dual_fusion": self.dual_fusion,
        }

    def score(self, item: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self.ascore_many([item]))[0]

    def score_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return asyncio.run(self.ascore_many(items))

    # ------------------------------------------------------------ async API
    async def ascore_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:

            async def one(idx: int, item: dict[str, Any]):
                async with sem:
                    return idx, await self._score_with_retry(client, item)

            results = await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
        out: list[dict[str, Any] | None] = [None] * len(items)
        for idx, res in results:
            out[idx] = res
        return out  # type: ignore[return-value]

    @staticmethod
    async def _acquire_two_phase() -> None:
        """Grab the process-wide two-phase permit without blocking this thread's
        event loop (each worker thread runs its own loop; a blocking acquire in
        one coroutine would freeze siblings that need to release permits)."""
        while not _TWO_PHASE_SEM.acquire(blocking=False):
            await asyncio.sleep(0.05)

    async def _score_with_retry(self, client: httpx.AsyncClient, item: dict[str, Any]) -> dict:
        endpoint = item.get("_endpoint") or "score"
        payload = {k: v for k, v in item.items() if not k.startswith("_")}
        two_phase = endpoint != "score"
        last_err: str = ""
        for attempt in range(RETRIES + 1):
            try:
                t0 = time.monotonic()
                if two_phase:
                    await self._acquire_two_phase()
                try:
                    r = await client.post(f"{self.base_url}/{endpoint}", json=payload)
                finally:
                    if two_phase:
                        _TWO_PHASE_SEM.release()
                r.raise_for_status()
                res = r.json()
                if res.get("status") == "success":
                    self._check_pin(res)
                    # client-side wall time incl. queueing (server compute is timing_s)
                    res["client_elapsed_s"] = round(time.monotonic() - t0, 4)
                    return res
                last_err = f"server:{res.get('error')}"
                break  # server-side scoring failure: not transient, don't retry
            except ProbeCheckpointChanged:
                raise
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_err = f"{type(exc).__name__}:{exc}"
            if attempt < RETRIES:
                await asyncio.sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
        failure = {
            "status": "failed",
            "error": last_err,
            "example_id": item.get("example_id"),
            "ts": time.time(),
        }
        self._log_failure(failure, item)
        return failure

    # ------------------------------------------------------------- helpers
    def _check_pin(self, res: dict[str, Any]) -> None:
        mode, ckpt = res.get("mode"), res.get("probe_checkpoint")
        if not self.pinned or mode not in self.pinned or not ckpt:
            return
        if ckpt == self.pinned[mode]:
            return
        # fused /score_context responses report "ctx_ckpt + dual_ckpt"
        if (mode == "context" and self.dual_fusion
                and ckpt == f"{self.pinned[mode]} + {self.dual_fusion}"):
            return
        raise ProbeCheckpointChanged(
            f"probe checkpoint changed mid-run for mode={mode}: "
            f"pinned={self.pinned[mode]} (dual={self.dual_fusion}) got={ckpt}"
        )

    def _log_failure(self, failure: dict, item: dict) -> None:
        if not self.failures_path:
            return
        rec = dict(failure)
        rec["item"] = {k: item.get(k) for k in ("example_id", "question", "mode")}
        self.failures_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.failures_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
