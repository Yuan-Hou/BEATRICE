#!/usr/bin/env python
"""Drive the token-span ablation end-to-end via the ctx_auto_scheduler machinery.

Stages (dependency-aware, one job per GPU, idempotent on log-success + output):
  1. 8 encoder retrains: spancq__{2 ctx branches} + qdrop{q,a}__{3 q-only branches}
     (drop-context / drop-question ctx encoders are reused from round-12 spanqa/spanca)
  2. 5 precomputes -> outputs/cached_fusion_vecs/span_ablation/<cond>/
  3. final merge + eval_subset + report via scripts/span_ablation_eval.py

Usage: setsid nohup python \
    scripts/run_span_ablation.py > logs/span_ablation/scheduler.out 2>&1 &
"""
from __future__ import annotations

import os
import subprocess
import sys

# gpu0=vLLM, gpu3/7=other users, gpu4=joint-k3 exhaustive, gpu2/5=occupied/forbidden
os.environ.setdefault("SCHED_EXCLUDE_GPUS", "0,2,3,4,5,7")

sys.path.insert(0, os.path.dirname(__file__))
import ctx_auto_scheduler as _cas  # noqa: E402
from ctx_auto_scheduler import PY, ROOT, Job, _init_pool, _pick_gpu, _release_gpu, schedule  # noqa: E402

# The 2k-context spancq trains peak well above the default 18GB gate under
# contention — that gate let jobs land on nearly-full shared cards and OOM
# crash-loop to FATAL (2026-07-11). Demand real headroom instead.
_cas.MIN_FREE_GB = 35.0

CACHE = "outputs/cached_fusion_vecs/span_ablation"
CTX_EVAL = "configs/experiment/probe/ctx_L17rd_attn_triples"
QONLY_EVAL = "configs/experiment/probe/L17rd_attn_triples"


def _eval_args(cfg_dir: str, suffix: str) -> list[str]:
    out = []
    for split in ("test", "popqa", "nq", "triviaqa"):
        out += ["--eval-config", f"{split}={cfg_dir}/eval_{split}_{suffix}.yaml"]
    return out


def main():
    _init_pool()
    jobs: dict[str, Job] = {}

    ctx_trains = [f"spancq__{b}" for b in ("hid_prob_attn", "hid_prob_resid")]
    for tag in ctx_trains:
        jobs[tag] = Job(
            tag, "train", f"configs/experiment/probe/ctx_ablation/train_{tag}.yaml",
            f"logs/ctx_ablation/{tag}.log", "Done. Best",
            f"outputs/ctx_ablation_probes/{tag}/context_checkpoint.pth")

    q_trains = {"qdropq": [], "qdropa": []}
    for variant in q_trains:
        for b in ("hid_resid", "hid_prob_attn", "hid_attn"):
            tag = f"{variant}__{b}"
            q_trains[variant].append(tag)
            jobs[tag] = Job(
                tag, "train", f"configs/experiment/probe/qonly_ablation/train_{tag}.yaml",
                f"logs/qonly_ablation/{tag}.log", "Done. Best",
                f"outputs/qonly_ablation_probes/{tag}/question_checkpoint.pth")

    pcs = [
        ("ctx_dropc", "context", [], _eval_args(CTX_EVAL, "gk2")),
        ("ctx_dropq", "context", [], _eval_args(CTX_EVAL, "gk2")),
        ("ctx_dropa", "context", ctx_trains, _eval_args(CTX_EVAL, "gk2")),
        ("qonly_dropq", "question", q_trains["qdropq"], _eval_args(QONLY_EVAL, "greedy_k3")),
        ("qonly_dropa", "question", q_trains["qdropa"], _eval_args(QONLY_EVAL, "greedy_k3")),
    ]
    for cond, side, deps, eargs in pcs:
        jobs[f"pc_{cond}"] = Job(
            f"pc_{cond}", "precompute",
            f"configs/experiment/probe/span_ablation/precompute_{cond}.yaml",
            f"logs/span_ablation/precompute_{cond}.log", "PRECOMPUTE_DONE",
            f"{CACHE}/{cond}/triviaqa.pt", deps=deps,
            cache_dir=f"{CACHE}/{cond}",
            extra_args=["--splits", "train,validation", "--side", side] + eargs)

    ok = schedule(jobs)
    if not ok:
        print("SPAN_ABLATION_FAILED (scheduler stage)")
        raise SystemExit(1)

    gpu = _pick_gpu()
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTORCH_NVML_BASED_CUDA_CHECK="0")
        log = ROOT / "logs/span_ablation/eval.log"
        with open(log, "w") as lf:
            r = subprocess.run([PY, "scripts/span_ablation_eval.py"],
                               stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    finally:
        _release_gpu(gpu)
    text = log.read_text(errors="ignore")
    if r.returncode == 0 and "SPAN_ABLATION_EVAL_DONE" in text:
        print("SPAN_ABLATION_DONE")
    else:
        print("SPAN_ABLATION_FAILED (eval stage) — tail:")
        print("\n".join(text.splitlines()[-8:]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
