"""Triple-modality (+ quad) encoders for the interaction-order pyramid (question-only).

Completes the 1/2/3/4-way modality-combination family: round 6 trained the 4 singles,
round 7 the 6 pairs; this generates the 4 triples (leave-one-modality-out early fusion)
and the 1 quad (all four modalities in one TokenResNet), same width/params as before
(use_attention_mass=true). Fusion/COMBINATION study happens afterwards on cached branch
vectors (scripts/search_fusion_combos.py), so only TRAIN configs are generated here.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CFG = Path("configs/experiment/probe")
SRC = CFG / "L17rd_attn"
OUT = CFG / "L17rd_attn_triples"
OUT.mkdir(parents=True, exist_ok=True)
TRIPLE_DIR = "outputs/L17rd_attn_triples_probes"

# numeric gate on when probability or residual is on (same convention as pairs)
COMBOS = {
    "hid_prob_resid":  dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=True,  use_residual_features=True,  use_attention_features=False),
    "hid_prob_attn":   dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=True,  use_residual_features=False, use_attention_features=True),
    "hid_resid_attn":  dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=False, use_residual_features=True,  use_attention_features=True),
    "prob_resid_attn": dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=True,  use_residual_features=True,  use_attention_features=True),
    "quad":            dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=True,  use_residual_features=True,  use_attention_features=True),
}


def load(p): return yaml.safe_load(open(p, encoding="utf-8"))
def dump(c, p): yaml.safe_dump(c, open(p, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)


tmpl = load(SRC / "train_only_hidden_L17rd_attn.yaml")
for name, flags in COMBOS.items():
    b = copy.deepcopy(tmpl)
    b["experiment_name"] = f"train_qa_resnet_triple_{name}_L17rd_attn"
    b["probe"].update(**flags)
    b["output"] = {"checkpoint_path": f"{TRIPLE_DIR}/{name}/question_checkpoint.pth",
                   "val_prediction_path": f"{TRIPLE_DIR}/{name}/question_val_preds.jsonl"}
    dump(b, OUT / f"train_triple_{name}.yaml")

print("wrote", len(list(OUT.glob("*.yaml"))), "configs to", OUT)
