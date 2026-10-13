"""Generate the round-12 ablation configs for the ctx champion (greedy_k2 branches).

Variants (all reuse the exact main-run training protocol; ablation = content zeroing
with dims/masks kept, so encoders/caches stay shape-compatible):

  Token-span ablation (both branches):
    spanqa : ablate_segments [context]            -> question+answer only
    spanca : ablate_segments [question]           -> context+answer only
    spana  : ablate_segments [context, question]  -> answer only
  Answer-budget ablation (both branches):
    ans8   : answer_max_len 8
    ans2   : answer_max_len 2
  Channel-level signal ablation:
    residnorm : hid_prob_resid, ablate_channels ["_lens"]  (norms only)
    residlens : hid_prob_resid, ablate_channels ["_norm"]  (lens only)
    attnent   : hid_prob_attn, use_attention_mass false    (entropy only)

Writes configs/experiment/probe/ctx_ablation/train_<variant>__<branch>.yaml
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

BASE = {
    "hid_prob_attn": "configs/experiment/probe/ctx_L17rd_attn_triples/train_triple_hid_prob_attn.yaml",
    "hid_prob_resid": "configs/experiment/probe/ctx_L17rd_attn_triples/train_triple_hid_prob_resid.yaml",
}
OUT_DIR = Path("configs/experiment/probe/ctx_ablation")
CKPT_ROOT = "outputs/ctx_ablation_probes"

VARIANTS = {  # name -> (branches, probe-key overrides)
    "spanqa": (["hid_prob_attn", "hid_prob_resid"], {"ablate_segments": ["context"]}),
    "spanca": (["hid_prob_attn", "hid_prob_resid"], {"ablate_segments": ["question"]}),
    "spana": (["hid_prob_attn", "hid_prob_resid"], {"ablate_segments": ["context", "question"]}),
    "ans8": (["hid_prob_attn", "hid_prob_resid"], {"answer_max_len": 8}),
    "ans2": (["hid_prob_attn", "hid_prob_resid"], {"answer_max_len": 2}),
    "residnorm": (["hid_prob_resid"], {"ablate_channels": ["_lens"]}),
    "residlens": (["hid_prob_resid"], {"ablate_channels": ["_norm"]}),
    "attnent": (["hid_prob_attn"], {"use_attention_mass": False}),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for variant, (branches, overrides) in VARIANTS.items():
        for branch in branches:
            cfg = copy.deepcopy(yaml.safe_load(open(BASE[branch])))
            tag = f"{variant}__{branch}"
            cfg["experiment_name"] = f"train_cqa_ablation_{tag}"
            cfg.pop("gpu_id", None)  # scheduler assigns
            cfg["probe"].update(overrides)
            cfg["output"] = {
                "checkpoint_path": f"{CKPT_ROOT}/{tag}/context_checkpoint.pth",
                "val_prediction_path": f"{CKPT_ROOT}/{tag}/context_val_preds.jsonl",
            }
            path = OUT_DIR / f"train_{tag}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            names.append(tag)
            print("wrote", path)
    print(f"{len(names)} configs")


if __name__ == "__main__":
    main()
