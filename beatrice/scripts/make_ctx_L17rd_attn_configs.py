"""Generate the with-context L17rd + attention-simplex 4-branch experiment (mirror of
scripts/make_L17rd_attn_configs.py for the context side).

Data = original-context (qwen3_5_9b_L17_rd, 4 files) + rewritten-context
(qwen3_5_9b_rewritten_context_L17_rd, 1 file) metas, both re-extracted with L17 hidden
and 64 resid_*_lens channels. use_attention_mass=true doubles the attention groups
(entropy + segment-mass simplex): withctx attention channels 256 -> 512. All four
single-signal branches retrain at that width, then full fusion + 4 leave-one-out
fusions + val/test/popqa evals.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CFG = Path("configs/experiment/probe")
OUT = CFG / "ctx_L17rd_attn"
OUT.mkdir(parents=True, exist_ok=True)
TEMPLATE = Path("outputs/signal_ablation_pairs/signal_ablation_v1/configs/only_hidden_train_context.yaml")
PROBE_DIR = "outputs/ctx_L17rd_attn_probes"

BRANCHES = {
    "only_hidden":      dict(use_last_hidden=True,  use_numeric_features=False, use_probability_features=False, use_residual_features=False, use_attention_features=False),
    "only_probability": dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=True,  use_residual_features=False, use_attention_features=False),
    "only_residual":    dict(use_last_hidden=False, use_numeric_features=True,  use_probability_features=False, use_residual_features=True,  use_attention_features=False),
    "only_attention":   dict(use_last_hidden=False, use_numeric_features=False, use_probability_features=False, use_residual_features=False, use_attention_features=True),
}
# GPUs 4 and 6 come free once the two feature extractions finish there.
BRANCH_GPU = {"only_hidden": "4", "only_probability": "4", "only_residual": "6", "only_attention": "6"}
FUSION_GPU = {"full": "4", "no_hidden": "4", "no_probability": "4", "no_residual": "6", "no_attention": "6"}


def load(p): return yaml.safe_load(open(p, encoding="utf-8"))
def dump(c, p): yaml.safe_dump(c, open(p, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
def repoint(paths):
    out = []
    for p in paths:
        p = p.replace("qwen3_5_9b_rewritten_context/", "qwen3_5_9b_rewritten_context_L17_rd/")
        # original-context dir (exact segment, avoid touching _question/_rewritten dirs)
        p = p.replace("features_judge_consensus/qwen3_5_9b/", "features_judge_consensus/qwen3_5_9b_L17_rd/")
        out.append(p)
    return out
def ckpt_of(branch): return f"{PROBE_DIR}/{branch}/context_checkpoint.pth"


# ---- 1. four single-signal branches at attn=512 ----
for branch, flags in BRANCHES.items():
    b = load(TEMPLATE)
    b["experiment_name"] = f"train_cqa_resnet_{branch}_withctx_ctxL17rd_attn"
    b["gpu_id"] = BRANCH_GPU[branch]
    for k in ("train_meta_paths", "validation_meta_paths"):
        b["data"][k] = repoint(b["data"][k])
    b["probe"].update(**flags, use_token_embedding=False, use_attention_mass=True)
    b["output"] = {"checkpoint_path": ckpt_of(branch),
                   "val_prediction_path": f"{PROBE_DIR}/{branch}/context_val_preds.jsonl"}
    b.pop("report", None)
    dump(b, OUT / f"train_{branch}_ctxL17rd_attn.yaml")

# ---- 2. full 4-branch fusion ----
f = load(CFG / "train_context_question_answer_frozen_fusion_withctx_qwen9b.yaml")
f["experiment_name"] = "train_cqa_frozen_fusion_withctx_ctxL17rd_attn"
f["gpu_id"] = FUSION_GPU["full"]
for k in ("train_meta_paths", "validation_meta_paths"):
    f["data"][k] = repoint(f["data"][k])
f["probe"]["use_attention_mass"] = True
f["probe"]["branch_checkpoints"] = [{"name": n, "checkpoint_path": ckpt_of(f"only_{n}")}
                                    for n in ("hidden", "probability", "residual", "attention")]
FUS = "outputs/checkpoints/frozen_fusion_context_question_answer_withctx_ctxL17rd_attn.pth"
f["output"] = {"checkpoint_path": FUS,
               "val_prediction_path": "outputs/predictions/frozen_fusion_context_withctx_ctxL17rd_attn_val.jsonl"}
dump(f, OUT / "train_fusion_ctxL17rd_attn.yaml")

# ---- 2b. leave-one-out fusions ----
fusion_ckpts = {"full": FUS}
for drop in ("hidden", "probability", "residual", "attention"):
    g = copy.deepcopy(f)
    g["experiment_name"] = f"train_cqa_frozen_fusion_withctx_ctxL17rd_attn_no_{drop}"
    g["gpu_id"] = FUSION_GPU[f"no_{drop}"]
    g["probe"]["branch_checkpoints"] = [bc for bc in f["probe"]["branch_checkpoints"] if bc["name"] != drop]
    ck = f"outputs/checkpoints/frozen_fusion_context_question_answer_withctx_ctxL17rd_attn_no_{drop}.pth"
    g["output"] = {"checkpoint_path": ck,
                   "val_prediction_path": f"outputs/predictions/frozen_fusion_context_withctx_ctxL17rd_attn_no_{drop}_val.jsonl"}
    dump(g, OUT / f"train_fusion_ctxL17rd_attn_no_{drop}.yaml")
    fusion_ckpts[f"no_{drop}"] = ck

# ---- 3. eval configs (validation/test/popqa) per fusion variant ----
for var, ck in fusion_ckpts.items():
    tag = "ctxL17rd_attn" if var == "full" else f"ctxL17rd_attn_{var}"
    for grp, src in [("test", "eval_frozen_fusion_context_test.yaml"),
                     ("popqa", "eval_frozen_fusion_context_popqa.yaml"),
                     ("validation", "eval_frozen_fusion_context_validation.yaml")]:
        e = load(CFG / src)
        e["experiment_name"] = f"eval_fusion_{grp}_{tag}"
        e["gpu_id"] = "4"
        e["data"]["meta_paths"] = repoint(e["data"]["meta_paths"])
        e["checkpoint_path"] = ck
        e["output"]["prediction_path"] = f"outputs/predictions/frozen_fusion_context_{grp}_{tag}_preds.jsonl"
        e["output"]["report_path"] = f"outputs/reports/frozen_fusion_context_{grp}_{tag}.html"
        dump(e, OUT / f"eval_{grp}_{tag}.yaml")

print("wrote", len(list(OUT.glob("*.yaml"))), "configs to", OUT)
