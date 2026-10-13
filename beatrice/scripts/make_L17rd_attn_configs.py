"""Generate the L17rd + attention-simplex 4-branch experiment.

Consolidates the residual-direction work into one clean 4-modality fusion:
  hidden@L17 + probability + residual(direction) + attention(symmetric simplex).

Symmetric attention: use_attention_mass=true stacks entropy + row-normalized mass
(attention_channels 128 -> 256). Because attention_channels is baked into every
branch's first conv, ALL four branches are retrained at 256 (hidden/prob/residual
just get 256-wide zeroed attention; only the attention branch reads real values).

Outputs: 4 branch trains + full fusion + 4 leave-one-out fusions + evals.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CFG = Path("configs/experiment/probe")
OUT = CFG / "L17rd_attn"
OUT.mkdir(parents=True, exist_ok=True)
RD = "qwen3_5_9b_question_L17_rd"                    # L17 hidden + resid-dir + orig attention
PROBE_DIR = "outputs/L17rd_attn_probes"

# per-branch single-signal flag sets (all use_attention_mass=true -> attn=256)
BRANCHES = {
    "only_hidden":      dict(use_last_hidden=True,  use_numeric_features=False, use_probability_features=False, use_residual_features=False, use_attention_features=False),
    "only_probability": dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=True,  use_residual_features=False, use_attention_features=False),
    "only_residual":    dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=False, use_residual_features=True,  use_attention_features=False),
    "only_attention":   dict(use_last_hidden=False, use_numeric_features=False, use_probability_features=False, use_residual_features=False, use_attention_features=True),
}
# GPUs 0/1 = user's vLLM; 2/5 saturated by other jobs; 4/6/7 mostly-allocated.
# Probes are small CNNs -> everything on GPU 3 (free), parallel jobs share it.
BRANCH_GPU = {"only_hidden": "3", "only_probability": "3", "only_residual": "3", "only_attention": "3"}
FUSION_NAME = {"hidden": "only_hidden", "probability": "only_probability", "residual": "only_residual", "attention": "only_attention"}
LOO_GPU = {"hidden": "3", "probability": "3", "residual": "3", "attention": "3"}


def load(p): return yaml.safe_load(open(p, encoding="utf-8"))
def dump(c, p): yaml.safe_dump(c, open(p, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
def repoint(paths):
    return [p.replace("qwen3_5_9b_question_L17", RD).replace("qwen3_5_9b_question", RD) for p in paths]
def ckpt_of(branch): return f"{PROBE_DIR}/{branch}/question_checkpoint.pth"


# ---- 1. four single-signal branches, all at attn=256 (use_attention_mass=true) ----
for branch, flags in BRANCHES.items():
    b = load(CFG / "train_question_answer_resnet_noctx_qwen9b.yaml")
    b["experiment_name"] = f"train_qa_resnet_{branch}_noctx_L17rd_attn"
    b["gpu_id"] = BRANCH_GPU[branch]
    for k in ("train_meta_paths", "validation_meta_paths"):
        b["data"][k] = repoint(b["data"][k])
    b["probe"].update(**flags, use_token_embedding=False, use_attention_mass=True)
    b["output"] = {"checkpoint_path": ckpt_of(branch),
                   "val_prediction_path": f"{PROBE_DIR}/{branch}/question_val_preds.jsonl"}
    dump(b, OUT / f"train_{branch}_L17rd_attn.yaml")

# ---- 2. full 4-branch fusion ----
f = load(CFG / "train_question_answer_frozen_fusion_noctx_qwen9b.yaml")
f["experiment_name"] = "train_qa_frozen_fusion_noctx_L17rd_attn"
f["gpu_id"] = "3"
for k in ("train_meta_paths", "validation_meta_paths"):
    f["data"][k] = repoint(f["data"][k])
f["probe"]["use_attention_mass"] = True
f["probe"]["branch_checkpoints"] = [{"name": n, "checkpoint_path": ckpt_of(FUSION_NAME[n])}
                                    for n in ("hidden", "probability", "residual", "attention")]
FUS = "outputs/checkpoints/frozen_fusion_question_answer_noctx_L17rd_attn.pth"
f["output"] = {"checkpoint_path": FUS,
               "val_prediction_path": "outputs/predictions/frozen_fusion_question_noctx_L17rd_attn_val.jsonl"}
dump(f, OUT / "train_fusion_L17rd_attn.yaml")

# ---- 2b. leave-one-out fusions (drop each modality) for the OOD ablation ----
fusion_ckpts = {"full": FUS}
for drop in ("hidden", "probability", "residual", "attention"):
    g = copy.deepcopy(f)
    g["experiment_name"] = f"train_qa_frozen_fusion_noctx_L17rd_attn_no_{drop}"
    g["gpu_id"] = LOO_GPU[drop]
    g["probe"]["branch_checkpoints"] = [bc for bc in f["probe"]["branch_checkpoints"] if bc["name"] != drop]
    ck = f"outputs/checkpoints/frozen_fusion_question_answer_noctx_L17rd_attn_no_{drop}.pth"
    g["output"] = {"checkpoint_path": ck,
                   "val_prediction_path": f"outputs/predictions/frozen_fusion_question_noctx_L17rd_attn_no_{drop}_val.jsonl"}
    dump(g, OUT / f"train_fusion_L17rd_attn_no_{drop}.yaml")
    fusion_ckpts[f"no_{drop}"] = ck

# ---- 3. eval configs (val/test/popqa) for each fusion variant ----
for var, ck in fusion_ckpts.items():
    tag = "L17rd_attn" if var == "full" else f"L17rd_attn_{var}"
    for grp, src in [("test", "eval_frozen_fusion_question_test.yaml"),
                     ("popqa", "eval_frozen_fusion_question_popqa.yaml"),
                     ("validation", "eval_frozen_fusion_question_validation.yaml")]:
        e = load(CFG / src)
        e["experiment_name"] = f"eval_fusion_{grp}_{tag}"
        e["gpu_id"] = "3"
        e["data"]["meta_paths"] = repoint(e["data"]["meta_paths"])
        e["checkpoint_path"] = ck
        e["output"]["prediction_path"] = f"outputs/predictions/frozen_fusion_question_{grp}_{tag}_preds.jsonl"
        e["output"]["report_path"] = f"outputs/reports/frozen_fusion_question_{grp}_{tag}.html"
        dump(e, OUT / f"eval_{grp}_{tag}.yaml")

print("wrote", len(list(OUT.glob("*.yaml"))), "configs to", OUT)
