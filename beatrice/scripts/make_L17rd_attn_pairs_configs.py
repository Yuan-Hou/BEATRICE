"""Pairwise-modality encoders as additional fusion branches (question-only).

Each single-signal branch currently convolves one modality with itself; the frozen
fusion head only sees pooled 128-d vectors, so token-level CROSS-modal interactions are
never modeled. This trains the 6 pairwise encoders (two modality flag-sets enabled at
once, same TokenResNet, same width as the singles: use_attention_mass=true), then two
fusion variants:
  - pairs6:          the 6 pair encoders only
  - singles4_pairs6: 4 existing singles (reused frozen) + 6 pairs = 10 branches
Compared against round-6 full4 (OOD .9133) and no_attention3 (OOD .9193). Also directly
tests whether attention's negative LOO was true redundancy or a late-fusion bottleneck
(watch hid_attn's synergy over the hidden single).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CFG = Path("configs/experiment/probe")
SRC = CFG / "L17rd_attn"
OUT = CFG / "L17rd_attn_pairs"
OUT.mkdir(parents=True, exist_ok=True)
PAIR_DIR = "outputs/L17rd_attn_pairs_probes"
SINGLE_DIR = "outputs/L17rd_attn_probes"
GPU = "3"

# flag-sets: numeric gate must be on when probability or residual is on
PAIRS = {
    "hid_prob":   dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=True,  use_residual_features=False, use_attention_features=False),
    "hid_resid":  dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=False, use_residual_features=True,  use_attention_features=False),
    "hid_attn":   dict(use_last_hidden=True,  use_numeric_features=False, use_probability_features=False, use_residual_features=False, use_attention_features=True),
    "prob_resid": dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=True,  use_residual_features=True,  use_attention_features=False),
    "prob_attn":  dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=True,  use_residual_features=False, use_attention_features=True),
    "resid_attn": dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=False, use_residual_features=True,  use_attention_features=True),
}
SINGLES = {n: f"{SINGLE_DIR}/only_{n}/question_checkpoint.pth"
           for n in ("hidden", "probability", "residual", "attention")}


def load(p): return yaml.safe_load(open(p, encoding="utf-8"))
def dump(c, p): yaml.safe_dump(c, open(p, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
def pair_ckpt(name): return f"{PAIR_DIR}/{name}/question_checkpoint.pth"


# ---- 1. six pairwise branches (template: round-6 branch config, keeps L17rd metas) ----
tmpl = load(SRC / "train_only_hidden_L17rd_attn.yaml")
for name, flags in PAIRS.items():
    b = copy.deepcopy(tmpl)
    b["experiment_name"] = f"train_qa_resnet_pair_{name}_noctx_L17rd_attn"
    b["gpu_id"] = GPU
    b["probe"].update(**flags)  # token_embedding/mass already set by template
    b["output"] = {"checkpoint_path": pair_ckpt(name),
                   "val_prediction_path": f"{PAIR_DIR}/{name}/question_val_preds.jsonl"}
    dump(b, OUT / f"train_pair_{name}.yaml")

# ---- 2. fusion variants ----
f0 = load(SRC / "train_fusion_L17rd_attn.yaml")
VARIANTS = {
    "pairs6": [{"name": n, "checkpoint_path": pair_ckpt(n)} for n in PAIRS],
    "singles4_pairs6": ([{"name": n, "checkpoint_path": p} for n, p in SINGLES.items()]
                        + [{"name": n, "checkpoint_path": pair_ckpt(n)} for n in PAIRS]),
}
fusion_ckpts = {}
for var, branches in VARIANTS.items():
    g = copy.deepcopy(f0)
    g["experiment_name"] = f"train_qa_frozen_fusion_noctx_L17rd_attn_{var}"
    g["gpu_id"] = GPU
    g["probe"]["branch_checkpoints"] = branches
    ck = f"outputs/checkpoints/frozen_fusion_question_answer_noctx_L17rd_attn_{var}.pth"
    g["output"] = {"checkpoint_path": ck,
                   "val_prediction_path": f"outputs/predictions/frozen_fusion_question_noctx_L17rd_attn_{var}_val.jsonl"}
    dump(g, OUT / f"train_fusion_{var}.yaml")
    fusion_ckpts[var] = ck

# ---- 3. evals ----
for var, ck in fusion_ckpts.items():
    for grp in ("validation", "test", "popqa"):
        e = load(SRC / f"eval_{grp}_L17rd_attn.yaml")
        e["experiment_name"] = f"eval_fusion_{grp}_L17rd_attn_{var}"
        e["gpu_id"] = GPU
        e["checkpoint_path"] = ck
        e["output"]["prediction_path"] = f"outputs/predictions/frozen_fusion_question_{grp}_L17rd_attn_{var}_preds.jsonl"
        e["output"]["report_path"] = f"outputs/reports/frozen_fusion_question_{grp}_L17rd_attn_{var}.html"
        dump(e, OUT / f"eval_{grp}_{var}.yaml")

print("wrote", len(list(OUT.glob("*.yaml"))), "configs to", OUT)
