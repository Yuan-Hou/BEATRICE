"""Unified runner: FlashRAG baselines + PGIR on the canonical-split datasets.

Methods (spec §5): norag, vanilla, ircot, iterretgen, flare, skr (dev
real-retrieval labels), selfask, sure, pgir. Adaptive-RAG needs a trained
query classifier; DRAGIN / SeaKR are stretch. (vanilla-pgir-template removed
2026-07-05 per user decision.)

Every run wraps the generator and retriever in counting proxies so the main
table gets #gen_calls / #retr_calls per item (§6-3); PGIR additionally logs
#score_calls in its trace.

Examples (run from <package_root>):
  python -m pgir.run_baselines --method vanilla --dataset hotpotqa --split test -n 200
  python -m pgir.run_baselines --method pgir --dataset musique --split dev --full-budget
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os as _os_runs

# PGIR_RUNS_ROOT isolates a campaign's outputs (e.g. runs/flashrag_gemma) so
# rebuild_tables' newest-wins over runs/flashrag never sees them.
RUNS = Path(_os_runs.environ.get(
    "PGIR_RUNS_ROOT", str(Path(__file__).resolve().parent / "runs" / "flashrag")))


class CountingGenerator:
    """Counts generate calls and records per-call wall time + batch size
    (call_log: [(n_prompts, elapsed_s), ...]) so per-request serial latency
    can be derived for baselines too."""

    def __init__(self, inner):
        self._inner = inner
        self.gen_calls = 0
        self.call_log: list[tuple[int, float]] = []

    def generate(self, input_list, *a, **kw):
        n = len(input_list) if isinstance(input_list, list) else 1
        self.gen_calls += n
        t0 = time.time()
        out = self._inner.generate(input_list, *a, **kw)
        self.call_log.append((n, round(time.time() - t0, 4)))
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class CountingRetriever:
    def __init__(self, inner):
        self._inner = inner
        self.retr_calls = 0

    def search(self, query, *a, **kw):
        self.retr_calls += 1
        return self._inner.search(query, *a, **kw)

    def batch_search(self, query, *a, **kw):
        n = len(query) if isinstance(query, list) else 1
        self.retr_calls += n
        return self._inner.batch_search(query, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


import os as _os_env

SKR_DATA = _os_env.environ.get(
    "PGIR_SKR_DATA",
    str(Path(__file__).resolve().parent / "assets" / "skr_training_data_devreal.json"))
# PGIR_ADAPTIVE_CLS points the adaptive judger at an alternative classifier dir
ADAPTIVE_CLS = _os_env.environ.get(
    "PGIR_ADAPTIVE_CLS",
    str(Path(__file__).resolve().parent / "assets" / "adaptive_classifier_t5large"))


def run_one(method: str, dataset_name: str, split: str = "test", sample_num=None,
            max_rounds: int = 4, topk: int = 3, tau_hi: float = 0.8, budget: int = 4,
            backend: str = "vllm", strategy: str = "s2", full_budget: bool = False,
            calibrator: str | None = None, workers: int = 8,
            save_note: str | None = None, flare_threshold: float = 0.2,
            skr_data: str = SKR_DATA, probing_theta: float = 0.0,
            bon_n: int = 1, bon_temperature: float = 0.8, bon_lo: float = 0.05,
            steer: bool = False, method_theta: float | None = None,
            tau0: float | None = None, patience: int | None = None) -> dict:
    """One (method, dataset, split) run. In-process so the 64GB index/corpus
    load once (`get_shared_retriever`) across many calls."""
    from flashrag.utils import get_dataset

    from pgir.flashrag_adapter import (PGIRFlashRAGPipeline, build_config,
                                       get_shared_retriever)
    from pgir.pipeline import PolicyCfg

    if method_theta is None:  # per-method sensible defaults (dev-tuned later)
        method_theta = {"dragin": 1.0, "seakr": 0.0}.get(method, 0.0)

    args = argparse.Namespace(
        method=method, dataset=dataset_name, split=split, sample_num=sample_num,
        max_rounds=max_rounds, topk=topk, tau_hi=tau_hi, budget=budget,
        backend=backend, strategy=strategy, full_budget=full_budget,
        calibrator=calibrator, workers=workers, save_note=save_note,
        flare_threshold=flare_threshold)

    note = args.save_note or f"{args.method}_{args.dataset}_{args.split}" + (
        f"_n{args.sample_num}" if args.sample_num else "")
    extra_cfg = {}
    if args.dataset in ("nq", "triviaqa"):  # native FlashRAG splits (generalization col)
        extra_cfg["data_dir"] = "pgir/hf_datasets/"
    if args.method == "adaptive":
        extra_cfg = {**extra_cfg, "judger_name": "adaptive",
                     "judger_config": {"model_path": ADAPTIVE_CLS}}
    if args.method == "skr":
        extra_cfg = {**extra_cfg, "judger_name": "skr",
                     "judger_config": {"model_path": "intfloat/e5-base-v2",
                                       "training_data_path": skr_data, "topk": 5}}
    import os as _os
    _rg = _os.environ.get("PGIR_RUN_GPU")
    if _rg:
        extra_cfg["gpu_id"] = str(_rg)
    config = build_config(
        args.dataset, args.split,
        test_sample_num=args.sample_num, retrieval_topk=args.topk,
        save_note=note, save_dir=str(RUNS / note), **extra_cfg)
    dataset = get_dataset(config)[args.split]

    # dragin/seakr load an HF model with a warmup forward, which MUST happen
    # before the 64GB faiss index occupies RAM (late cublasCreate ALLOC_FAILED)
    hf_runner = None
    if args.method in ("dragin", "seakr"):
        from pgir import MODEL_NAME, VLLM_URL
        if args.method == "dragin":
            from pgir.dragin import DRAGIN
            hf_runner = DRAGIN(MODEL_NAME, VLLM_URL, None,
                               theta=method_theta, max_rounds=args.max_rounds - 1,
                               k_docs=args.topk, record_signals=args.full_budget)
        else:
            from pgir.seakr import SeaKR
            # k_docs / max_rounds use SeaKR's OWN faithful defaults (max_docs=5,
            # max_reasoning_steps=7), not the shared topk — they are intrinsic
            # to the method (official repo values).
            hf_runner = SeaKR(MODEL_NAME, VLLM_URL, None,
                              theta=method_theta, record_u=args.full_budget)

    retriever = CountingRetriever(get_shared_retriever(config))
    if hf_runner is not None:
        hf_runner.retriever = retriever

    t0 = time.time()
    if args.method == "pgir":
        pipe = PGIRFlashRAGPipeline(
            config, policy=PolicyCfg(tau_hi=args.tau_hi, budget=args.budget, k=args.topk,
                                     bon_n=bon_n, bon_temperature=bon_temperature,
                                     bon_lo=bon_lo, steer=steer,
                                     tau0=tau0, patience=patience),
            backend=args.backend, strategy=args.strategy, full_budget=args.full_budget,
            calibrator_path=args.calibrator, retriever=retriever, workers=args.workers)
        result = pipe.run(dataset)
        gen_calls = None  # PGIR counts per-item cost in its trace
    else:
        from flashrag.pipeline import (FLAREPipeline, IRCOTPipeline,
                                       IterativePipeline, SequentialPipeline)
        from flashrag.utils import get_generator

        generator = CountingGenerator(get_generator(config))
        if args.method in ("norag", "vanilla"):
            pipe = SequentialPipeline(config, retriever=retriever, generator=generator)
            result = (pipe.naive_run(dataset) if args.method == "norag"
                      else pipe.run(dataset))
        elif args.method == "ircot":
            pipe = IRCOTPipeline(config, retriever=retriever, generator=generator,
                                 max_iter=args.max_rounds)
            result = pipe.run(dataset)
        elif args.method == "iterretgen":
            pipe = IterativePipeline(config, retriever=retriever, generator=generator,
                                     iter_num=args.max_rounds)
            result = pipe.run(dataset)
        elif args.method == "flare":
            pipe = FLAREPipeline(config, threshold=args.flare_threshold,
                                 max_iter_num=args.max_rounds,
                                 retriever=retriever, generator=generator)
            result = pipe.run(dataset)
        elif args.method == "skr":
            from flashrag.pipeline import ConditionalPipeline
            pipe = ConditionalPipeline(config, retriever=retriever, generator=generator)
            result = pipe.run(dataset)
        elif args.method == "selfask":
            from flashrag.pipeline import SelfAskPipeline
            pipe = SelfAskPipeline(config, max_iter=args.max_rounds,
                                   retriever=retriever, generator=generator)
            result = pipe.run(dataset)
        elif args.method == "sure":
            from flashrag.pipeline.branching_pipeline import SuRePipeline
            pipe = SuRePipeline(config, retriever=retriever, generator=generator)
            result = pipe.run(dataset)
        elif args.method == "adaptive":
            from flashrag.pipeline import AdaptivePipeline
            pipe = AdaptivePipeline(config, retriever=retriever, generator=generator)
            result = pipe.run(dataset)
        elif args.method == "probingrag":
            from flashrag.pipeline import BasicPipeline

            from pgir import MODEL_NAME, VLLM_URL
            from pgir.probingrag.pipeline import ProbingRAG

            pr = ProbingRAG(MODEL_NAME, VLLM_URL, retriever,
                            theta=probing_theta, max_iter=args.max_rounds,
                            k=args.topk, record_all_rounds=args.full_budget)
            pr_items = [{"example_id": it.id, "question": it.question} for it in dataset]
            traces = pr.run_items(pr_items)
            tdir = Path(config["save_dir"])
            tdir.mkdir(parents=True, exist_ok=True)
            with open(tdir / "probingrag_trace.jsonl", "w") as f:
                for tr in traces:
                    f.write(json.dumps(tr, ensure_ascii=False) + "\n")
            dataset.update_output("pred", [tr["answer"] for tr in traces])
            base = BasicPipeline(config)
            result = base.evaluate(dataset, do_eval=True)
            generator.gen_calls = sum(tr["cost"]["gen_calls"] for tr in traces)
            retriever.retr_calls = sum(tr["cost"]["retr_calls"] for tr in traces)
        elif args.method in ("dragin", "seakr"):
            from flashrag.pipeline import BasicPipeline

            runner = hf_runner  # constructed before the index load (see above)
            mr_items = [{"example_id": it.id, "question": it.question} for it in dataset]
            traces = runner.run_items(mr_items)
            tdir = Path(config["save_dir"])
            tdir.mkdir(parents=True, exist_ok=True)
            with open(tdir / f"{args.method}_trace.jsonl", "w") as f:
                for tr in traces:
                    f.write(json.dumps(tr, ensure_ascii=False) + "\n")
            dataset.update_output("pred", [tr["answer"] for tr in traces])
            base = BasicPipeline(config)
            result = base.evaluate(dataset, do_eval=True)
            generator.gen_calls = sum(tr["cost"]["gen_calls"] for tr in traces)
            retriever.retr_calls = sum(tr["cost"]["retr_calls"] for tr in traces)
        gen_calls = generator.gen_calls

    n_items = len(result)
    summary = {
        "method": args.method, "dataset": args.dataset, "split": args.split,
        "n": n_items, "wall_s": round(time.time() - t0, 1),
        "gen_calls_per_item": (gen_calls / n_items) if gen_calls is not None else None,
        "retr_calls_per_item": retriever.retr_calls / n_items,
    }
    out = Path(config["save_dir"]) / "cost_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    if args.method != "pgir":  # PGIR's per-call timing lives in its trace
        gen_log = [{"n_prompts": n, "elapsed_s": s} for n, s in generator.call_log]
        (Path(config["save_dir"]) / "gen_call_log.jsonl").write_text(
            "\n".join(json.dumps(r) for r in gen_log) + ("\n" if gen_log else ""))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-theta", type=float, default=None,
                    help="dragin RIND / seakr uncertainty threshold "
                         "(default: dragin 1.0, seakr 0.0)")
    ap.add_argument("--method", required=True,
                    choices=["norag", "vanilla", "ircot", "iterretgen", "flare",
                             "skr", "selfask", "sure", "adaptive", "probingrag",
                             "dragin", "seakr", "pgir"])
    ap.add_argument("--probing-theta", type=float, default=0.0)
    ap.add_argument("--flare-threshold", type=float, default=0.2)
    ap.add_argument("--dataset", required=True,
                    choices=["hotpotqa", "twowiki", "musique", "popqa", "nq", "triviaqa",
                             "nq_ood", "triviaqa_ood"])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("-n", "--sample-num", type=int, default=None,
                    help="subset size (smoke); default full split")
    ap.add_argument("--max-rounds", type=int, default=4, help="iteration cap (§6-3)")
    ap.add_argument("--topk", type=int, default=3)
    # pgir-specific
    ap.add_argument("--tau-hi", type=float, default=0.8)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "scorer"])
    ap.add_argument("--strategy", default="s2", choices=["s1", "s2", "s3"])
    ap.add_argument("--full-budget", action="store_true",
                    help="no early accept (val trace collection)")
    ap.add_argument("--calibrator", default=None, help="path to calibrator json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-note", default=None)
    ap.add_argument("--bon-n", type=int, default=1,
                    help="best-of-N candidates per round incl. greedy (1 = off)")
    ap.add_argument("--bon-temp", type=float, default=0.8)
    ap.add_argument("--bon-lo", type=float, default=0.05,
                    help="lower edge of the sampling band [bon_lo, tau_hi)")
    ap.add_argument("--tau0", type=float, default=None,
                    help="round-0 accept threshold (closed-book waiver)")
    ap.add_argument("--patience", type=int, default=None,
                    help="stop after N stagnant rounds -> pool argmax")
    ap.add_argument("--steer", action="store_true",
                    help="probe-steered evidence actions (s1/rewrite/s2)")
    a = ap.parse_args()
    run_one(a.method, a.dataset, a.split, a.sample_num, a.max_rounds, a.topk,
            a.tau_hi, a.budget, a.backend, a.strategy, a.full_budget,
            a.calibrator, a.workers, a.save_note, flare_threshold=a.flare_threshold,
            probing_theta=a.probing_theta, bon_n=a.bon_n, bon_temperature=a.bon_temp,
            bon_lo=a.bon_lo, steer=a.steer, method_theta=a.method_theta,
            tau0=a.tau0, patience=a.patience)


if __name__ == "__main__":
    main()
