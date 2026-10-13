"""Train the 5 withctx triple/quad encoders via the ctx_auto_scheduler machinery,
restricted to GPUs disjoint from the heavy lanes (default eligible: 2,5,6 —
gpu0=vLLM, gpu1=service+ctx-precompute, gpu3/4/7=other users right now).

Usage:  python scripts/run_ctx_triples.py
"""
from __future__ import annotations

import os

os.environ.setdefault("SCHED_EXCLUDE_GPUS", "0,1,3,4,7")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from ctx_auto_scheduler import Job, _init_pool, schedule  # noqa: E402

BRANCHES = ["hid_prob_resid", "hid_prob_attn", "hid_resid_attn", "prob_resid_attn", "quad"]


def main():
    _init_pool()
    jobs = {}
    for b in BRANCHES:
        jobs[b] = Job(
            b, "train",
            f"configs/experiment/probe/ctx_L17rd_attn_triples/train_triple_{b}.yaml",
            f"logs/ctx_L17rd_attn_triples/branch_{b}.log", "Done. Best",
            f"outputs/ctx_L17rd_attn_triples_probes/{b}/context_checkpoint.pth")
    ok = schedule(jobs)
    print("CTX_TRIPLES_DONE" if ok else "CTX_TRIPLES_FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
