"""Train the 13 round-12 ablation encoders (token-span / answer-budget / channel-level)
via the ctx_auto_scheduler machinery on GPUs 1,2,5,6 (gpu0=vLLM, gpu3/7=other users,
gpu4=round-11 finisher).

Usage:  setsid nohup python scripts/run_ctx_ablation.py > logs/ctx_ablation/scheduler.out 2>&1 &
"""
from __future__ import annotations

import os

os.environ.setdefault("SCHED_EXCLUDE_GPUS", "0,3,7")  # use gpu1/2/4/5/6 (gpu4 freed after lookback)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from ctx_auto_scheduler import Job, _init_pool, schedule  # noqa: E402

# Round-12 relaunch: the first wave (spanqa/spanca x2) is DONE — best ckpts kept on disk
# (.9134-.9247), micro-improvement under patience-10 was noise (< seed variance). Remaining
# 9 encoders re-run under capped configs (epochs=15, patience=3, min_delta=0.001) so they
# terminate in a bounded time instead of chasing +.002 at ~2h/epoch for days.
TAGS = [
    "spana__hid_prob_attn", "spana__hid_prob_resid",
    "ans8__hid_prob_attn", "ans8__hid_prob_resid",
    "ans2__hid_prob_attn", "ans2__hid_prob_resid",
    "residnorm__hid_prob_resid", "residlens__hid_prob_resid",
    "attnent__hid_prob_attn",
]


def main():
    _init_pool()
    jobs = {}
    for tag in TAGS:
        jobs[tag] = Job(
            tag, "train",
            f"configs/experiment/probe/ctx_ablation/train_{tag}.yaml",
            f"logs/ctx_ablation/{tag}.log", "Done. Best",
            f"outputs/ctx_ablation_probes/{tag}/context_checkpoint.pth")
    ok = schedule(jobs)
    print("CTX_ABLATION_DONE" if ok else "CTX_ABLATION_FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
