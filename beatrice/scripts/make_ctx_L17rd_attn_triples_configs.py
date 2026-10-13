"""With-context triple/quad encoders (mirror of make_L17rd_attn_triples_configs.py).

Completes the ctx interaction-order pyramid: round-8 singles + round-9 pairs + these
4 triples + 1 quad. Combination search then runs on the merged 15-branch vec cache.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CFG = Path("configs/experiment/probe")
SRC = CFG / "ctx_L17rd_attn"
OUT = CFG / "ctx_L17rd_attn_triples"
OUT.mkdir(parents=True, exist_ok=True)
TRIPLE_DIR = "outputs/ctx_L17rd_attn_triples_probes"

COMBOS = {
    "hid_prob_resid":  dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=True,  use_residual_features=True,  use_attention_features=False),
    "hid_prob_attn":   dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=True,  use_residual_features=False, use_attention_features=True),
    "hid_resid_attn":  dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=False, use_residual_features=True,  use_attention_features=True),
    "prob_resid_attn": dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=True,  use_residual_features=True,  use_attention_features=True),
    "quad":            dict(use_last_hidden=True,  use_numeric_features=True,  use_probability_features=True,  use_residual_features=True,  use_attention_features=True),
}


def load(p): return yaml.safe_load(open(p, encoding="utf-8"))
def dump(c, p): yaml.safe_dump(c, open(p, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)


tmpl = load(SRC / "train_only_hidden_ctxL17rd_attn.yaml")
for name, flags in COMBOS.items():
    b = copy.deepcopy(tmpl)
    b["experiment_name"] = f"train_cqa_resnet_triple_{name}_withctx_ctxL17rd_attn"
    b["probe"].update(**flags)
    b["output"] = {"checkpoint_path": f"{TRIPLE_DIR}/{name}/context_checkpoint.pth",
                   "val_prediction_path": f"{TRIPLE_DIR}/{name}/context_val_preds.jsonl"}
    dump(b, OUT / f"train_triple_{name}.yaml")

print("wrote", len(list(OUT.glob("*.yaml"))), "configs to", OUT)
