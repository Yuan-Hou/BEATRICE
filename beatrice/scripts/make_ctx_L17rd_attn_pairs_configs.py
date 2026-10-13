"""With-context pairwise-modality encoder experiment (mirror of
scripts/make_L17rd_attn_pairs_configs.py for the context side).

Six pairwise encoders (two modality flag-sets enabled at once, same
ContextQuestionAnswerResNet, same width as the singles: use_attention_mass=true ->
attn=512), then fusions:
  - pairs6:          the 6 pair encoders only
  - singles4_pairs6: 4 ctx singles (reused frozen) + 6 pairs = 10 branches
Compared against ctx full4 / no_* from the ctx_L17rd_attn experiment.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CFG = Path("configs/experiment/probe")
SRC = CFG / "ctx_L17rd_attn"
OUT = CFG / "ctx_L17rd_attn_pairs"
OUT.mkdir(parents=True, exist_ok=True)
PAIR_DIR = "outputs/ctx_L17rd_attn_pairs_probes"
SINGLE_DIR = "outputs/ctx_L17rd_attn_probes"

PAIRS = {
    "hid_prob":   dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=True,  use_residual_features=False, use_attention_features=False),
    "hid_resid":  dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=False, use_residual_features=True,  use_attention_features=False),
    "hid_attn":   dict(use_last_hidden=True,  use_numeric_features=False, use_probability_features=False, use_residual_features=False, use_attention_features=True),
    "prob_resid": dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=True,  use_residual_features=True,  use_attention_features=False),
    "prob_attn":  dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=True,  use_residual_features=False, use_attention_features=True),
    "resid_attn": dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=False, use_residual_features=True,  use_attention_features=True),
}
PAIR_GPU = {"hid_prob": "3", "hid_resid": "3", "hid_attn": "4", "prob_resid": "4", "prob_attn": "6", "resid_attn": "6"}
SINGLES = {n: f"{SINGLE_DIR}/only_{n}/context_checkpoint.pth"
           for n in ("hidden", "probability", "residual", "attention")}


def load(p): return yaml.safe_load(open(p, encoding="utf-8"))
def dump(c, p): yaml.safe_dump(c, open(p, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
def pair_ckpt(name): return f"{PAIR_DIR}/{name}/context_checkpoint.pth"


# ---- 1. six pairwise branches ----
tmpl = load(SRC / "train_only_hidden_ctxL17rd_attn.yaml")
for name, flags in PAIRS.items():
    b = copy.deepcopy(tmpl)
    b["experiment_name"] = f"train_cqa_resnet_pair_{name}_withctx_ctxL17rd_attn"
    b["gpu_id"] = PAIR_GPU[name]
    b["probe"].update(**flags)  # token_embedding/mass already set by template
    b["output"] = {"checkpoint_path": pair_ckpt(name),
                   "val_prediction_path": f"{PAIR_DIR}/{name}/context_val_preds.jsonl"}
    dump(b, OUT / f"train_pair_{name}.yaml")

# ---- 2. fusion variants ----
f0 = load(SRC / "train_fusion_ctxL17rd_attn.yaml")
VARIANTS = {
    "pairs6": [{"name": n, "checkpoint_path": pair_ckpt(n)} for n in PAIRS],
    "singles4_pairs6": ([{"name": n, "checkpoint_path": p} for n, p in SINGLES.items()]
                        + [{"name": n, "checkpoint_path": pair_ckpt(n)} for n in PAIRS]),
}
fusion_ckpts = {}
for var, branches in VARIANTS.items():
    g = copy.deepcopy(f0)
    g["experiment_name"] = f"train_cqa_frozen_fusion_withctx_ctxL17rd_attn_{var}"
    g["gpu_id"] = "4"
    g["probe"]["branch_checkpoints"] = branches
    ck = f"outputs/checkpoints/frozen_fusion_context_question_answer_withctx_ctxL17rd_attn_{var}.pth"
    g["output"] = {"checkpoint_path": ck,
                   "val_prediction_path": f"outputs/predictions/frozen_fusion_context_withctx_ctxL17rd_attn_{var}_val.jsonl"}
    dump(g, OUT / f"train_fusion_{var}.yaml")
    fusion_ckpts[var] = ck

# ---- 3. evals ----
for var, ck in fusion_ckpts.items():
    for grp in ("validation", "test", "popqa"):
        e = load(SRC / f"eval_{grp}_ctxL17rd_attn.yaml")
        e["experiment_name"] = f"eval_fusion_{grp}_ctxL17rd_attn_{var}"
        e["gpu_id"] = "4"
        e["checkpoint_path"] = ck
        e["output"]["prediction_path"] = f"outputs/predictions/frozen_fusion_context_{grp}_ctxL17rd_attn_{var}_preds.jsonl"
        e["output"]["report_path"] = f"outputs/reports/frozen_fusion_context_{grp}_ctxL17rd_attn_{var}.html"
        dump(e, OUT / f"eval_{grp}_{var}.yaml")

print("wrote", len(list(OUT.glob("*.yaml"))), "configs to", OUT)
