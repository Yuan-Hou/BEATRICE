"""Generate the gemma search-stage precompute configs: per side one FULL
15-branch frozen-fusion config (for precompute_frozen_fusion_vecs) plus the
four eval-split data configs (test_id / popqa / nq / triviaqa), mirroring the
qwen round-12 pipeline. Branch names match the qwen search vocabulary
(hidden .. quad) so downstream combo tooling is reusable as-is.

Outputs under configs/experiment/probe/gemma4_search/.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "configs/experiment/probe"
OUT = PROBE / "gemma4_search"

MODEL = "google/gemma-4-E4B-it"
Q_DIR = "data/gemma4/features/gemma4_question_L26_rd"
C_DIR = "data/gemma4/features/gemma4_L26_rd"
IDS = "configs/sampling/gemma4_scope_ids.json"

CTX_TRAIN = [f"{C_DIR}/{d}_shard{i}of2_features_meta_ok.jsonl"
             for d in ("musique", "hotpotqa_supporting_only", "twowiki_supporting_only") for i in range(2)]
CTX_REWR = [f"{C_DIR}/rewritten_context_shard{i}of4_features_meta_ok.jsonl" for i in range(4)]
Q_TRAIN = [f"{Q_DIR}/{d}_question_features_meta_ok.jsonl"
           for d in ("musique", "hotpotqa_supporting_only", "twowiki_supporting_only")]

BRANCHES = [
    ("hidden", "{p}_probes/only_hidden"),
    ("probability", "{p}_probes/only_probability"),
    ("residual", "{p}_probes/only_residual"),
    ("attention", "{p}_probes/only_attention"),
    ("hid_prob", "{p}_pairs_probes/hid_prob"),
    ("hid_resid", "{p}_pairs_probes/hid_resid"),
    ("hid_attn", "{p}_pairs_probes/hid_attn"),
    ("prob_resid", "{p}_pairs_probes/prob_resid"),
    ("prob_attn", "{p}_pairs_probes/prob_attn"),
    ("resid_attn", "{p}_pairs_probes/resid_attn"),
    ("hid_prob_resid", "{p}_triples_probes/hid_prob_resid"),
    ("hid_prob_attn", "{p}_triples_probes/hid_prob_attn"),
    ("hid_resid_attn", "{p}_triples_probes/hid_resid_attn"),
    ("prob_resid_attn", "{p}_triples_probes/prob_resid_attn"),
    ("quad", "{p}_triples_probes/quad"),
]


def branch_ckpts(side: str) -> list[dict]:
    prefix = "outputs/gemma4/ctx_L17rd_attn" if side == "context" else "outputs/gemma4/L17rd_attn"
    fname = "context_checkpoint.pth" if side == "context" else "question_checkpoint.pth"
    return [{"name": n, "checkpoint_path": f"{tpl.format(p=prefix)}/{fname}"} for n, tpl in BRANCHES]


def partition(group: str, source: str, name: str) -> dict:
    return {"enabled": True, "source_splits": [source], "ids_path": IDS, "group": group, "name": name}


def make_full(side: str) -> Path:
    tpl = (PROBE / "ctx_L17rd_attn/train_fusion_ctxL17rd_attn.yaml" if side == "context"
           else PROBE / "L17rd_attn/train_fusion_L17rd_attn.yaml")
    cfg = yaml.safe_load(tpl.open())
    cfg["experiment_name"] = f"gemma4_{cfg['experiment_name']}_all15"
    cfg["model_name"] = MODEL
    cfg.pop("gpu_id", None)
    d = cfg["data"]
    d["train_meta_paths"] = (CTX_TRAIN + CTX_REWR) if side == "context" else list(Q_TRAIN)
    d["validation_meta_paths"] = list(CTX_TRAIN) if side == "context" else list(Q_TRAIN)
    d.pop("judge_consensus_purified", None)
    d.pop("purification_summary_path", None)
    d["train_partition"] = partition("train", "train", "gemma4_train")
    d["validation_partition"] = partition("validation", "validation", "gemma4_validation")
    cfg["probe"]["branch_checkpoints"] = branch_ckpts(side)
    o = cfg.get("output", {})
    for k in list(o):
        if isinstance(o[k], str):
            o[k] = o[k].replace("outputs/", "outputs/gemma4/", 1)
    path = OUT / f"pc_{'ctx' if side == 'context' else 'q'}_all15.yaml"
    yaml.safe_dump(cfg, path.open("w"), sort_keys=False)
    return path


def make_evals(side: str) -> None:
    fam_dir = C_DIR if side == "context" else Q_DIR
    sfx = "_features_meta_ok.jsonl" if side == "context" else "_question_features_meta_ok.jsonl"
    tag = "ctx" if side == "context" else "q"
    sets = {
        "test": {
            "meta_paths": CTX_TRAIN if side == "context" else Q_TRAIN,
            "splits": ["validation"],
            "partition": partition("test", "validation", "gemma4_test"),
        },
        "popqa": {"meta_paths": [f"{fam_dir}/popqa{sfx}"], "splits": ["test"]},
        "nq": {"meta_paths": [f"{fam_dir}/nq{sfx}"], "splits": ["validation"]},
        "triviaqa": {"meta_paths": [f"{fam_dir}/triviaqa{sfx}"], "splits": ["validation"]},
    }
    for name, data in sets.items():
        yaml.safe_dump({"experiment_name": f"gemma4_pc_eval_{tag}_{name}", "data": data},
                       (OUT / f"eval_{tag}_{name}.yaml").open("w"), sort_keys=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for side in ("context", "question"):
        p = make_full(side)
        make_evals(side)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
