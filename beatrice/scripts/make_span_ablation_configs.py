#!/usr/bin/env python
"""Generate configs for the token-span ablation (drop exactly ONE of question /
answer / context) under the NEW joint-champion setting.

Encoder retrains (ablate_segments zeroes the segment's numeric/hidden/attention
content at TRAIN time; masks/dims kept, so the frozen-fusion cache pipeline works
unchanged — the ckpt round-trips the ablation into precompute):

  ctx side (base = the two ctx champion branches; drop-context/drop-question
  encoders already exist from round-12 as spanqa/spanca):
    spancq__{hid_prob_attn,hid_prob_resid} : ablate [answer]     (NEW)
  q-only side (base = the q-only champion branches, hid_attn doubles as the
  joint champion's closed-book branch):
    qdropq__{hid_resid,hid_prob_attn,hid_attn} : ablate [question]
    qdropa__{hid_resid,hid_prob_attn,hid_attn} : ablate [answer]

Precompute configs (fusion-style, branch_checkpoints -> span-ablated ckpts):
    span_ablation/precompute_{ctx_dropc,ctx_dropq,ctx_dropa,qonly_dropq,qonly_dropa}.yaml

Run order: this script -> scripts/run_span_ablation.py (scheduler) ->
scripts/span_ablation_eval.py (merge + eval_subset + report).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CTX_ABL = Path("configs/experiment/probe/ctx_ablation")
QONLY_ABL = Path("configs/experiment/probe/qonly_ablation")
SPAN = Path("configs/experiment/probe/span_ablation")

QONLY_BASE = {
    "hid_resid": "configs/experiment/probe/L17rd_attn_pairs/train_pair_hid_resid.yaml",
    "hid_attn": "configs/experiment/probe/L17rd_attn_pairs/train_pair_hid_attn.yaml",
    "hid_prob_attn": "configs/experiment/probe/L17rd_attn_triples/train_triple_hid_prob_attn.yaml",
}
CTX_FUSION_BASE = "configs/experiment/probe/ctx_L17rd_attn_triples/train_fusion_greedy_k2.yaml"
QONLY_FUSION_BASE = "configs/experiment/probe/L17rd_attn_triples/train_fusion_greedy_k3.yaml"

CTX_CKPT = "outputs/ctx_ablation_probes/{tag}/context_checkpoint.pth"
QONLY_CKPT = "outputs/qonly_ablation_probes/{tag}/question_checkpoint.pth"


def _load(p):
    return yaml.safe_load(open(p, encoding="utf-8"))


def _dump(cfg, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(p, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    print(f"wrote {p}")


def main():
    # ---- ctx drop-answer encoders (reuse the capped round-12 protocol configs) ----
    for branch in ("hid_prob_attn", "hid_prob_resid"):
        cfg = copy.deepcopy(_load(CTX_ABL / f"train_spanca__{branch}.yaml"))
        tag = f"spancq__{branch}"
        cfg["experiment_name"] = f"train_cqa_ablation_{tag}"
        cfg.pop("gpu_id", None)
        cfg["probe"]["ablate_segments"] = ["answer"]
        cfg["output"] = {
            "checkpoint_path": CTX_CKPT.format(tag=tag),
            "val_prediction_path": f"outputs/ctx_ablation_probes/{tag}/context_val_preds.jsonl",
        }
        _dump(cfg, CTX_ABL / f"train_{tag}.yaml")

    # ---- q-only span-drop encoders ----
    for variant, seg in (("qdropq", "question"), ("qdropa", "answer")):
        for branch, base in QONLY_BASE.items():
            cfg = copy.deepcopy(_load(base))
            tag = f"{variant}__{branch}"
            cfg["experiment_name"] = f"train_qa_ablation_{tag}"
            cfg.pop("gpu_id", None)
            cfg["probe"]["ablate_segments"] = [seg]
            cfg["output"] = {
                "checkpoint_path": QONLY_CKPT.format(tag=tag),
                "val_prediction_path": f"outputs/qonly_ablation_probes/{tag}/question_val_preds.jsonl",
            }
            _dump(cfg, QONLY_ABL / f"train_{tag}.yaml")

    # ---- precompute fusion configs ----
    ctx_src = {  # condition -> encoder tag prefix providing the two ctx branches
        "ctx_dropc": "spanqa",   # ablate [context]  (round-12 ckpts, reused)
        "ctx_dropq": "spanca",   # ablate [question] (round-12 ckpts, reused)
        "ctx_dropa": "spancq",   # ablate [answer]   (new)
    }
    for cond, prefix in ctx_src.items():
        cfg = copy.deepcopy(_load(CTX_FUSION_BASE))
        cfg["experiment_name"] = f"precompute_span_{cond}"
        cfg.pop("gpu_id", None)
        for bc in cfg["probe"]["branch_checkpoints"]:
            bc["checkpoint_path"] = CTX_CKPT.format(tag=f"{prefix}__{bc['name']}")
        cfg["output"]["checkpoint_path"] = f"outputs/checkpoints/_unused_span_{cond}.pth"
        _dump(cfg, SPAN / f"precompute_{cond}.yaml")

    for cond, variant in (("qonly_dropq", "qdropq"), ("qonly_dropa", "qdropa")):
        cfg = copy.deepcopy(_load(QONLY_FUSION_BASE))
        cfg["experiment_name"] = f"precompute_span_{cond}"
        cfg.pop("gpu_id", None)
        names = [bc["name"] for bc in cfg["probe"]["branch_checkpoints"]]
        assert sorted(names) == sorted(QONLY_BASE), names
        for bc in cfg["probe"]["branch_checkpoints"]:
            bc["checkpoint_path"] = QONLY_CKPT.format(tag=f"{variant}__{bc['name']}")
        cfg["output"]["checkpoint_path"] = f"outputs/checkpoints/_unused_span_{cond}.pth"
        _dump(cfg, SPAN / f"precompute_{cond}.yaml")


if __name__ == "__main__":
    main()
