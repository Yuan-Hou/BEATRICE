"""Train the 5 Q-only triple/quad encoders using the ctx_auto_scheduler machinery,
restricted to a GPU set disjoint from the running withctx scheduler (default 2,5,7).

Usage:  SCHED_EXCLUDE_GPUS=0,1,3,4,6 python scripts/run_qonly_triples.py
"""
from __future__ import annotations

import os

os.environ.setdefault("SCHED_EXCLUDE_GPUS", "0,1,3,4,6")

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
            f"configs/experiment/probe/L17rd_attn_triples/train_triple_{b}.yaml",
            f"logs/L17rd_attn_triples/branch_{b}.log", "Done. Best",
            f"outputs/L17rd_attn_triples_probes/{b}/question_checkpoint.pth")
    ok = schedule(jobs)
    print("QONLY_TRIPLES_DONE" if ok else "QONLY_TRIPLES_FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
