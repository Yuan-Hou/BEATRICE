"""FlashRAG <-> PGIR glue.

- `build_config(...)`: FlashRAG Config from our base yaml + overrides.
- `RetrieverAdapter`: FlashRAG retriever behind PGIR's `search(query, num)`.
- `PGIRFlashRAGPipeline(BasicPipeline)`: runs the PGIR loop over a FlashRAG
  dataset (items concurrently in a thread pool; the services batch), writes a
  per-item trace jsonl + run metadata (probe checkpoint fingerprints), and
  evaluates EM/F1 with FlashRAG's evaluator like every baseline.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flashrag.config import Config
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever

from pgir import MODEL_NAME, PROBE_SCORER_URL, VLLM_URL
from pgir.arm import ArmEvaluator
from pgir.calibration import PerModeCalibrator
from pgir.pipeline import PGIR, PolicyCfg
from pgir.scorer_client import ScorerClient
from pgir.vllm_client import VLLMClient

# PGIR_FLASHRAG_YAML swaps the whole host config (backbone migration, e.g.
# flashrag_gemma.yaml); default untouched so existing runs stay reproducible.
BASE_YAML = os.environ.get(
    "PGIR_FLASHRAG_YAML",
    str(Path(__file__).resolve().parent / "configs" / "flashrag_qwen.yaml"))


def build_config(dataset_name: str, split: str = "test", **overrides) -> Config:
    cfg = {"dataset_name": dataset_name, "split": [split]}
    cfg.update(overrides)
    return Config(BASE_YAML, cfg)


_RETRIEVER_CACHE: dict = {}


def get_shared_retriever(config: Config):
    """One retriever per (index, corpus) — the Flat index is ~60GB RAM."""
    key = (config["index_path"], config["corpus_path"])
    if key not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[key] = get_retriever(config)
    return _RETRIEVER_CACHE[key]


class RetrieverAdapter:
    """All FlashRAG retriever work happens on ONE dedicated thread.

    Calling faiss/torch-encoder/HF-datasets from many worker threads crashed
    the PGIR smoke twice: first a deadlock at the first retrieval round
    (faiss OpenMP), then a segfault from concurrent lazy C-extension imports
    (pandas/numpy via datasets' load_docs). A single-thread executor pins every
    retrieval to the same thread; retrieval is fast vs 1-2s gen/score calls,
    so serialisation costs nothing."""

    def __init__(self, flashrag_retriever, warmup: bool = True):
        self.r = flashrag_retriever
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="pgir-retrieval")
        try:  # cap faiss's OpenMP fan-out (see pgir/__init__ BLAS note)
            import faiss
            faiss.omp_set_num_threads(min(32, os.cpu_count() or 32))
        except Exception:
            pass
        if warmup:  # exercise encoder + faiss + corpus lazy imports once
            self.search("warmup query", 1)

    def search(self, query: str, num: int) -> list[dict]:
        docs = self._pool.submit(self.r.search, query, num=num).result()
        return docs or []


class PGIRFlashRAGPipeline(BasicPipeline):
    def __init__(
        self,
        config: Config,
        policy: PolicyCfg | None = None,
        backend: str = "vllm",
        strategy: str = "s2",
        full_budget: bool = False,
        calibrator_path: str | None = None,
        retriever=None,
        workers: int = 8,
        scorer_url: str = PROBE_SCORER_URL,
        vllm_url: str = VLLM_URL,
    ):
        super().__init__(config)
        self.policy = policy or PolicyCfg(tau_hi=0.8, budget=4, k=config["retrieval_topk"])
        self.workers = int(workers)
        self.run_dir = Path(config["save_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.retriever = retriever if retriever is not None else get_shared_retriever(config)
        scorer = ScorerClient(scorer_url, failures_path=self.run_dir / "scorer_failures.jsonl")
        vllm = VLLMClient(vllm_url, MODEL_NAME) if backend == "vllm" else None
        calibrator = PerModeCalibrator.load(calibrator_path) if calibrator_path else None
        if calibrator is not None:
            # fail loud if this calibrator was fit against a different probe than
            # the scorer now serves (silent p_cal corruption otherwise). With a
            # dual q+ctx head, fused /score_context arms carry the fingerprint
            # "ctx_ckpt + dual_ckpt" — a calibrator fit on such traces must be
            # checked against the same composite, not the bare ctx checkpoint.
            pinned = dict(scorer.pinned)
            if scorer.dual_fusion and "context" in pinned:
                pinned["context"] = f"{pinned['context']} + {scorer.dual_fusion}"
            calibrator.check_provenance(pinned, strict=True)
        evaluator = ArmEvaluator(scorer, backend=backend, vllm=vllm, calibrator=calibrator)
        self.pgir = PGIR(evaluator, RetrieverAdapter(self.retriever), self.policy,
                         strategy=strategy, full_budget=full_budget)

        self.run_metadata = {
            **scorer.run_metadata(),
            "vllm_url": vllm_url, "backend": backend, "strategy": strategy,
            "full_budget": full_budget, "calibrator": calibrator_path,
            "policy": {"tau_hi": self.policy.tau_hi, "tau_lo": self.policy.tau_lo,
                       "budget": self.policy.budget, "k": self.policy.k},
            "dataset": config["dataset_name"], "split": list(config["split"]),
            "seed": config["seed"], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.run_dir / "pgir_run_metadata.json").write_text(
            json.dumps(self.run_metadata, indent=2))

    def run(self, dataset, do_eval: bool = True, pred_process_fun=None,
            trace_path: str | None = None):
        items = list(dataset)

        def one(item):
            return self.pgir.run_item(item.question, example_id=item.id)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(one, items))

        trace_file = Path(trace_path) if trace_path else (
            self.run_dir / f"pgir_trace_{self.config['dataset_name']}"
                           f"_{'-'.join(self.config['split'])}.jsonl")
        with open(trace_file, "w") as f:
            for res in results:
                f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")

        dataset.update_output("pred", [r.answer for r in results])
        dataset.update_output("pgir_action", [r.action for r in results])
        dataset.update_output("pgir_stop_round", [r.stop_round for r in results])
        dataset.update_output("pgir_cost", [r.cost.to_dict() for r in results])
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
