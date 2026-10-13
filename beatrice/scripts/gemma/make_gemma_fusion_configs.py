"""Generate gemma frozen-fusion training configs for the service champions.

Subsets = each setting's greedy-k3 result (2026-07-18); if the exhaustive k3
champion differs, regenerate with the new subset and retrain (one swap).
Derives from the all15 precompute configs (same data bindings), keeping the
qwen fusion probe hyperparameters (epochs 60 / patience 10).
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "configs/experiment/probe/gemma4_search"

SETS = {
    "ctx": {
        "template": S / "pc_ctx_all15.yaml",
        "branches": ["hid_prob", "hidden", "attention"],
        "ckpt": "outputs/gemma4/checkpoints/frozen_fusion_context_gemma4_greedy_k3.pth",
        "preds": "outputs/gemma4/predictions/frozen_fusion_context_gemma4_greedy_k3_val.jsonl",
    },
    "q": {
        "template": S / "pc_q_all15.yaml",
        "branches": ["quad", "hid_prob_attn", "hidden"],
        "ckpt": "outputs/gemma4/checkpoints/frozen_fusion_question_gemma4_greedy_k3.pth",
        "preds": "outputs/gemma4/predictions/frozen_fusion_question_gemma4_greedy_k3_val.jsonl",
    },
}


def main():
    for tag, spec in SETS.items():
        cfg = yaml.safe_load(spec["template"].open())
        cfg["experiment_name"] = f"gemma4_fusion_{tag}_greedy_k3"
        keep = set(spec["branches"])
        bc = [b for b in cfg["probe"]["branch_checkpoints"] if b["name"] in keep]
        assert len(bc) == 3, (tag, [b["name"] for b in bc])
        cfg["probe"]["branch_checkpoints"] = bc
        cfg["output"] = {"checkpoint_path": spec["ckpt"], "val_prediction_path": spec["preds"]}
        cfg.pop("report", None)
        out = S / f"train_fusion_{tag}_greedy_k3.yaml"
        yaml.safe_dump(cfg, out.open("w"), sort_keys=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
