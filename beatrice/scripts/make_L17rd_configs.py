"""Generate L17_rd configs: L17 hidden + directional-residual features. Retrain the
only_residual branch on the augmented tensors, then a fusion that uses the new residual
branch + the L17 hidden branch + the original probability branch. Also leave-one-out
fusion configs (drop residual / drop hidden) for the OOD marginal-contribution ablation.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CFG = Path("configs/experiment/probe")
OUT = CFG / "L17rd"
OUT.mkdir(parents=True, exist_ok=True)
RD = "qwen3_5_9b_question_L17_rd"          # augmented metas (L17 hidden + resid-dir)
HID_CKPT = "outputs/L17_probes/only_hidden/question_checkpoint.pth"      # reuse L17 hidden branch
RESID_CKPT = "outputs/L17rd_probes/only_residual/question_checkpoint.pth"
PROB_CKPT = "outputs/signal_ablation_pairs/signal_ablation_v1/artifacts/only_probability/question_checkpoint.pth"


def load(p): return yaml.safe_load(open(p, encoding="utf-8"))
def dump(c, p): yaml.safe_dump(c, open(p, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
def rd(paths): return [p.replace("qwen3_5_9b_question_L17", RD).replace("qwen3_5_9b_question", RD)
                       if "L17_rd" not in p else p for p in paths]


def repoint(paths):
    out = []
    for p in paths:
        # original metas live in qwen3_5_9b_question; send everything to the _rd root
        out.append(p.replace("qwen3_5_9b_question_L17", RD).replace("qwen3_5_9b_question", RD))
    return out


# ---- 1. only_residual branch on augmented (directional) residual features ----
b = load(CFG / "train_question_answer_resnet_noctx_qwen9b.yaml")
b["experiment_name"] = "train_qa_resnet_only_residual_noctx_L17rd"
b["gpu_id"] = "2"
for k in ("train_meta_paths", "validation_meta_paths"):
    b["data"][k] = repoint(b["data"][k])
b["probe"].update(use_last_hidden=False, use_numeric_features=True, use_probability_features=False,
                  use_residual_features=True, use_attention_features=False, use_token_embedding=False)
b["output"] = {"checkpoint_path": RESID_CKPT,
               "val_prediction_path": "outputs/L17rd_probes/only_residual/question_val_preds.jsonl"}
dump(b, OUT / "train_only_residual_L17rd.yaml")

# ---- 2. fusion: hidden(L17) + residual(dir) + probability ----
f = load(CFG / "train_question_answer_frozen_fusion_noctx_qwen9b.yaml")
f["experiment_name"] = "train_qa_frozen_fusion_noctx_L17rd"
f["gpu_id"] = "2"
for k in ("train_meta_paths", "validation_meta_paths"):
    f["data"][k] = repoint(f["data"][k])
f["probe"]["branch_checkpoints"] = [
    {"name": "hidden", "checkpoint_path": HID_CKPT},
    {"name": "probability", "checkpoint_path": PROB_CKPT},
    {"name": "residual", "checkpoint_path": RESID_CKPT},
]
FUS = "outputs/checkpoints/frozen_fusion_question_answer_noctx_L17rd.pth"
f["output"] = {"checkpoint_path": FUS,
               "val_prediction_path": "outputs/predictions/frozen_fusion_question_noctx_L17rd_val.jsonl"}
dump(f, OUT / "train_fusion_L17rd.yaml")

# ---- 2b. leave-one-out fusions for the ablation (drop residual; drop hidden) ----
LOO_GPU = {"residual": "3", "hidden": "5"}
for drop in ("residual", "hidden"):
    g = copy.deepcopy(f)
    g["experiment_name"] = f"train_qa_frozen_fusion_noctx_L17rd_no_{drop}"
    g["gpu_id"] = LOO_GPU[drop]
    g["probe"]["branch_checkpoints"] = [bc for bc in f["probe"]["branch_checkpoints"] if bc["name"] != drop]
    ck = f"outputs/checkpoints/frozen_fusion_question_answer_noctx_L17rd_no_{drop}.pth"
    g["output"] = {"checkpoint_path": ck,
                   "val_prediction_path": f"outputs/predictions/frozen_fusion_question_noctx_L17rd_no_{drop}_val.jsonl"}
    dump(g, OUT / f"train_fusion_L17rd_no_{drop}.yaml")

# ---- 3. eval configs for each fusion variant ----
variants = {"L17rd": FUS,
            "L17rd_no_residual": "outputs/checkpoints/frozen_fusion_question_answer_noctx_L17rd_no_residual.pth",
            "L17rd_no_hidden": "outputs/checkpoints/frozen_fusion_question_answer_noctx_L17rd_no_hidden.pth"}
for var, ck in variants.items():
    for grp, src in [("test", "eval_frozen_fusion_question_test.yaml"),
                     ("popqa", "eval_frozen_fusion_question_popqa.yaml"),
                     ("validation", "eval_frozen_fusion_question_validation.yaml")]:
        e = load(CFG / src)
        e["experiment_name"] = f"eval_fusion_{grp}_{var}"
        e["gpu_id"] = "2"
        e["data"]["meta_paths"] = repoint(e["data"]["meta_paths"])
        e["checkpoint_path"] = ck
        e["output"]["prediction_path"] = f"outputs/predictions/frozen_fusion_question_{grp}_{var}_preds.jsonl"
        e["output"]["report_path"] = f"outputs/reports/frozen_fusion_question_{grp}_{var}.html"
        dump(e, OUT / f"eval_{grp}_{var}.yaml")

print("wrote", len(list(OUT.glob("*.yaml"))), "configs to", OUT)
