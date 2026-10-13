#!/usr/bin/env python
"""Temporal-receptive-field ablation: pointwise (kernel_size=1) branch encoders.

The ONLY ablated variable is the residual blocks' conv kernel (3 -> 1, padding 1 -> 0),
i.e. cross-token temporal mixing vs strictly per-token processing (a per-token MLP +
the same masked max||mean pooling). Channels, hidden_dim, num_blocks, norms, dropout,
pooling, fusion head, training recipe (the champions' own configs, verbatim) and the
3-seed eval_subset protocol are all identical. (Param count necessarily shrinks with
the kernel — that is inherent to the ablation, footnoted in the report.)

Retrains the champion branch encoders of BOTH settings with conv_kernel_size=1:
  ctx:   hid_prob_attn, hid_prob_resid           (joint champion's context branches)
  qonly: hid_resid, hid_prob_attn, hid_attn      (q-only champion; hid_attn doubles as
                                                  the joint champion's closed-book branch)
then precomputes frozen vecs and evaluates the same champion subsets.

Writes outputs/reports/convk1_ablation.{json,md}.
Usage: setsid nohup python scripts/run_convk1_ablation.py > logs/convk1/driver.out 2>&1 &
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

os.environ.setdefault("SCHED_EXCLUDE_GPUS", "1,2,3,4,5,7")  # gpu0 + gpu6 (gpu4 = span chain)

sys.path.insert(0, os.path.dirname(__file__))
import ctx_auto_scheduler as _cas  # noqa: E402
from ctx_auto_scheduler import PY, ROOT, Job, _init_pool, _pick_gpu, _release_gpu, schedule  # noqa: E402

_cas.MIN_FREE_GB = 35.0  # 2k-ctx trains OOM-loop on nearly-full shared cards (span-ablation lesson)

CTX_BASE = {b: f"configs/experiment/probe/ctx_L17rd_attn_triples/train_triple_{b}.yaml"
            for b in ("hid_prob_attn", "hid_prob_resid")}
Q_BASE = {
    "hid_resid": "configs/experiment/probe/L17rd_attn_pairs/train_pair_hid_resid.yaml",
    "hid_attn": "configs/experiment/probe/L17rd_attn_pairs/train_pair_hid_attn.yaml",
    "hid_prob_attn": "configs/experiment/probe/L17rd_attn_triples/train_triple_hid_prob_attn.yaml",
}
CFG_DIR = Path("configs/experiment/probe/convk1")
CKPT = "outputs/convk1_probes/{tag}/{side}_checkpoint.pth"
CACHE = "outputs/cached_fusion_vecs/convk1"
CTX_EVAL = "configs/experiment/probe/ctx_L17rd_attn_triples"
Q_EVAL = "configs/experiment/probe/L17rd_attn_triples"


def gen_configs():
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    for side, bases, fus_base in (
        ("context", CTX_BASE, f"{CTX_EVAL}/train_fusion_greedy_k2.yaml"),
        ("question", Q_BASE, f"{Q_EVAL}/train_fusion_greedy_k3.yaml"),
    ):
        for b, base in bases.items():
            cfg = copy.deepcopy(yaml.safe_load(open(base)))
            tag = f"convk1__{b}" + ("_q" if side == "question" else "")
            cfg["experiment_name"] = f"train_{tag}"
            cfg.pop("gpu_id", None)
            cfg["probe"]["conv_kernel_size"] = 1
            cfg["output"] = {
                "checkpoint_path": CKPT.format(tag=tag, side=side),
                "val_prediction_path": f"outputs/convk1_probes/{tag}/{side}_val_preds.jsonl",
            }
            yaml.safe_dump(cfg, open(CFG_DIR / f"train_{tag}.yaml", "w"), sort_keys=False, allow_unicode=True)
        pc = copy.deepcopy(yaml.safe_load(open(fus_base)))
        cond = "ctx" if side == "context" else "qonly"
        pc["experiment_name"] = f"precompute_convk1_{cond}"
        pc.pop("gpu_id", None)
        for bc in pc["probe"]["branch_checkpoints"]:
            tag = f"convk1__{bc['name']}" + ("_q" if side == "question" else "")
            bc["checkpoint_path"] = CKPT.format(tag=tag, side=side)
        pc["output"]["checkpoint_path"] = f"outputs/checkpoints/_unused_convk1_{cond}.pth"
        yaml.safe_dump(pc, open(CFG_DIR / f"precompute_{cond}.yaml", "w"), sort_keys=False, allow_unicode=True)


def main():
    gen_configs()
    _init_pool()
    jobs: dict[str, Job] = {}
    ctx_tags, q_tags = [], []
    for b in CTX_BASE:
        tag = f"convk1__{b}"
        ctx_tags.append(tag)
        jobs[tag] = Job(tag, "train", f"{CFG_DIR}/train_{tag}.yaml",
                        f"logs/convk1/{tag}.log", "Done. Best",
                        CKPT.format(tag=tag, side="context"))
    for b in Q_BASE:
        tag = f"convk1__{b}_q"
        q_tags.append(tag)
        jobs[tag] = Job(tag, "train", f"{CFG_DIR}/train_{tag}.yaml",
                        f"logs/convk1/{tag}.log", "Done. Best",
                        CKPT.format(tag=tag, side="question"))

    def eargs(cfg_dir, suffix):
        out = []
        for split in ("test", "popqa", "nq", "triviaqa"):
            out += ["--eval-config", f"{split}={cfg_dir}/eval_{split}_{suffix}.yaml"]
        return out

    jobs["pc_ctx"] = Job("pc_ctx", "precompute", f"{CFG_DIR}/precompute_ctx.yaml",
                         "logs/convk1/precompute_ctx.log", "PRECOMPUTE_DONE",
                         f"{CACHE}/ctx/triviaqa.pt", deps=ctx_tags, cache_dir=f"{CACHE}/ctx",
                         extra_args=["--splits", "train,validation", "--side", "context"] + eargs(CTX_EVAL, "gk2"))
    jobs["pc_qonly"] = Job("pc_qonly", "precompute", f"{CFG_DIR}/precompute_qonly.yaml",
                           "logs/convk1/precompute_qonly.log", "PRECOMPUTE_DONE",
                           f"{CACHE}/qonly/triviaqa.pt", deps=q_tags, cache_dir=f"{CACHE}/qonly",
                           extra_args=["--splits", "train,validation", "--side", "question"] + eargs(Q_EVAL, "greedy_k3"))

    ok = schedule(jobs)
    if not ok:
        print("CONVK1_FAILED (scheduler stage)")
        raise SystemExit(1)

    gpu = _pick_gpu()
    try:
        code = r'''
import json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from pathlib import Path
from search_fusion_combos import eval_subset, load_split
from span_ablation_eval import ALL_SPLITS, CTX_CHAMP, EVAL_SPLITS, QONLY_CHAMP, merged_ctx_split
CV = Path("outputs/cached_fusion_vecs")
out = {"note": "conv_kernel_size 3->1 in every branch-encoder residual block; only the temporal receptive field is ablated"}
runs = [
    ("withctx_full", "context", CTX_CHAMP, lambda: {s: load_split(CV/"joint_30br", s, "cuda") for s in ALL_SPLITS}),
    ("withctx_k1", "context", CTX_CHAMP, lambda: {s: merged_ctx_split(CV/"convk1/ctx", CV/"convk1/qonly", s) for s in ALL_SPLITS}),
    ("qonly_full", "question", QONLY_CHAMP, lambda: {s: load_split(CV/"qonly_all15", s, "cuda") for s in ALL_SPLITS}),
    ("qonly_k1", "question", QONLY_CHAMP, lambda: {s: load_split(CV/"convk1/qonly", s, "cuda") for s in ALL_SPLITS}),
]
for name, side, champ, loader in runs:
    sp = loader()
    r = eval_subset(champ, sp, side, "cuda", [0, 1, 2], eval_splits=EVAL_SPLITS)
    out[name] = {k: v for k, v in r.items() if isinstance(v, float)}
    print(f"[{name}] val={r['val']:.4f} " + " ".join(f"{s}={r[s]:.4f}" for s in EVAL_SPLITS), flush=True)
    del sp
json.dump(out, open("outputs/reports/convk1_ablation.json", "w"), indent=1)
lines = ["# Conv kernel ablation: k=3 (champion) vs k=1 (pointwise / per-token MLP)", "",
         "Only the residual blocks' temporal receptive field is ablated; dims/norm/dropout/",
         "pooling/fusion/training recipe/3-seed protocol identical (param count shrinks with",
         "the kernel; inherent to the ablation).", "",
         "| setting | variant | val | " + " | ".join(EVAL_SPLITS) + " |",
         "|---|---|---|" + "---|" * len(EVAL_SPLITS)]
for name in ("withctx_full", "withctx_k1", "qonly_full", "qonly_k1"):
    r = out[name]
    lines.append("| " + name.replace("_", " | ", 1) + " | " + f"{r['val']:.4f} | " +
                 " | ".join(f"{r[s]:.4f}" for s in EVAL_SPLITS) + " |")
Path("outputs/reports/convk1_ablation.md").write_text("\n".join(lines) + "\n")
print("CONVK1_EVAL_DONE")
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_NVML_BASED_CUDA_CHECK="0")
        r = subprocess.run([PY, "-c", code], env=env, cwd=ROOT)
    finally:
        _release_gpu(gpu)
    print("CONVK1_DONE" if r.returncode == 0 else "CONVK1_FAILED (eval stage)")
    raise SystemExit(0 if r.returncode == 0 else 1)


if __name__ == "__main__":
    main()
