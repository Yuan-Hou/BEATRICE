#!/usr/bin/env python
"""Re-run the q-only drop-question span-ablation encoders (2 extra independent
draws, s1/s2), precompute their caches, and print all draws side by side so a
clean, representative run can be picked for the presentation table.

(Encoder training has no explicit seed; independent runs differ through unseeded
RNG + cudnn nondeterminism, which is exactly the run-to-run variance we sample.)

Usage: SCHED_EXCLUDE_GPUS="1,2,3,4,5,7" setsid nohup python \
    scripts/rerun_span_qonly_dropq.py > logs/span_ablation/rerun_dropq.out 2>&1 &
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

os.environ.setdefault("SCHED_EXCLUDE_GPUS", "1,2,3,4,5,7")  # allow gpu0 + gpu6

sys.path.insert(0, os.path.dirname(__file__))
from ctx_auto_scheduler import PY, ROOT, Job, _init_pool, _pick_gpu, _release_gpu, schedule  # noqa: E402

BRANCHES = ["hid_resid", "hid_prob_attn", "hid_attn"]
RUNS = ["s1", "s2"]
QCFG = Path("configs/experiment/probe/qonly_ablation")
SPAN = Path("configs/experiment/probe/span_ablation")
CACHE = "outputs/cached_fusion_vecs/span_ablation"


def gen_configs():
    for run in RUNS:
        for b in BRANCHES:
            cfg = copy.deepcopy(yaml.safe_load(open(QCFG / f"train_qdropq__{b}.yaml")))
            tag = f"qdropq__{b}_{run}"
            cfg["experiment_name"] = f"train_qa_ablation_{tag}"
            cfg.pop("gpu_id", None)
            cfg["output"] = {
                "checkpoint_path": f"outputs/qonly_ablation_probes/{tag}/question_checkpoint.pth",
                "val_prediction_path": f"outputs/qonly_ablation_probes/{tag}/question_val_preds.jsonl",
            }
            yaml.safe_dump(cfg, open(QCFG / f"train_{tag}.yaml", "w"), sort_keys=False, allow_unicode=True)
        pc = copy.deepcopy(yaml.safe_load(open(SPAN / "precompute_qonly_dropq.yaml")))
        pc["experiment_name"] = f"precompute_span_qonly_dropq_{run}"
        for bc in pc["probe"]["branch_checkpoints"]:
            bc["checkpoint_path"] = f"outputs/qonly_ablation_probes/qdropq__{bc['name']}_{run}/question_checkpoint.pth"
        yaml.safe_dump(pc, open(SPAN / f"precompute_qonly_dropq_{run}.yaml", "w"), sort_keys=False, allow_unicode=True)


def main():
    gen_configs()
    _init_pool()
    jobs: dict[str, Job] = {}
    ev = ["--eval-config", "test=configs/experiment/probe/L17rd_attn_triples/eval_test_greedy_k3.yaml",
          "--eval-config", "popqa=configs/experiment/probe/L17rd_attn_triples/eval_popqa_greedy_k3.yaml",
          "--eval-config", "nq=configs/experiment/probe/L17rd_attn_triples/eval_nq_greedy_k3.yaml",
          "--eval-config", "triviaqa=configs/experiment/probe/L17rd_attn_triples/eval_triviaqa_greedy_k3.yaml"]
    for run in RUNS:
        tags = []
        for b in BRANCHES:
            tag = f"qdropq__{b}_{run}"
            tags.append(tag)
            jobs[tag] = Job(
                tag, "train", f"{QCFG}/train_{tag}.yaml",
                f"logs/qonly_ablation/{tag}.log", "Done. Best",
                f"outputs/qonly_ablation_probes/{tag}/question_checkpoint.pth")
        jobs[f"pc_{run}"] = Job(
            f"pc_{run}", "precompute", f"{SPAN}/precompute_qonly_dropq_{run}.yaml",
            f"logs/span_ablation/precompute_qonly_dropq_{run}.log", "PRECOMPUTE_DONE",
            f"{CACHE}/qonly_dropq_{run}/triviaqa.pt", deps=tags,
            cache_dir=f"{CACHE}/qonly_dropq_{run}",
            extra_args=["--splits", "train,validation", "--side", "question"] + ev)
    ok = schedule(jobs)
    if not ok:
        print("RERUN_DROPQ_FAILED")
        raise SystemExit(1)

    # ---- side-by-side eval of all draws ----
    gpu = _pick_gpu()
    try:
        code = r'''
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
import json
from search_fusion_combos import eval_subset, load_split
from span_ablation_eval import ALL_SPLITS, EVAL_SPLITS, QONLY_CHAMP
CV = Path("outputs/cached_fusion_vecs")
out = {}
for name, cache in [("full", CV / "qonly_all15"), ("dropq_s0", CV / "span_ablation/qonly_dropq"),
                    ("dropq_s1", CV / "span_ablation/qonly_dropq_s1"),
                    ("dropq_s2", CV / "span_ablation/qonly_dropq_s2")]:
    sp = {s: load_split(cache, s, "cuda") for s in ALL_SPLITS}
    r = eval_subset(QONLY_CHAMP, sp, "question", "cuda", [0, 1, 2], eval_splits=EVAL_SPLITS)
    out[name] = {k: r[k] for k in r if isinstance(r[k], float)}
    print(f"[{name}] val={r['val']:.4f} " + " ".join(f"{s}={r[s]:.4f}" for s in EVAL_SPLITS), flush=True)
    del sp
json.dump(out, open("outputs/reports/span_qonly_dropq_draws.json", "w"), indent=1)
print("DRAWS_DONE")
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_NVML_BASED_CUDA_CHECK="0")
        r = subprocess.run([PY, "-c", code], env=env, cwd=ROOT)
    finally:
        _release_gpu(gpu)
    print("RERUN_DROPQ_DONE" if r.returncode == 0 else "RERUN_DROPQ_FAILED")


if __name__ == "__main__":
    main()
