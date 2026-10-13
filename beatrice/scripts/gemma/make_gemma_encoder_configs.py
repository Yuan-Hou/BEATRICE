"""Generate the full 30-encoder training config set for gemma (-it, L26).

Mirrors the qwen encoder campaign 1:1 — per setting 4 singles + 6 pairs +
4 triples + 1 quad — by loading each qwen train config as the template and
rewriting only the model/data/output bindings. Probe hyperparameters stay
byte-identical to the paper protocol.

Outputs:
  configs/experiment/probe/gemma4_encoders/{q,ctx}/*.yaml
  configs/experiment/probe/gemma4_encoders/manifest.json  (driver input)
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "configs/experiment/probe"
OUT = PROBE / "gemma4_encoders"

MODEL = "google/gemma-4-E4B-it"
Q_DIR = "data/gemma4/features/gemma4_question_L26_rd"
C_DIR = "data/gemma4/features/gemma4_L26_rd"

# qwen meta basename -> gemma meta_ok path list (ctx datasets are sharded)
META_MAP = {
    "musique_features_meta.jsonl": [f"{C_DIR}/musique_shard{i}of2_features_meta_ok.jsonl" for i in range(2)],
    "hotpotqa_supporting_only_features_meta.jsonl": [f"{C_DIR}/hotpotqa_supporting_only_shard{i}of2_features_meta_ok.jsonl" for i in range(2)],
    "twowiki_supporting_only_features_meta.jsonl": [f"{C_DIR}/twowiki_supporting_only_shard{i}of2_features_meta_ok.jsonl" for i in range(2)],
    "rewritten_context_cqa_features_meta.jsonl": [f"{C_DIR}/rewritten_context_shard{i}of4_features_meta_ok.jsonl" for i in range(4)],
    "musique_question_features_meta.jsonl": [f"{Q_DIR}/musique_question_features_meta_ok.jsonl"],
    "hotpotqa_supporting_only_question_features_meta.jsonl": [f"{Q_DIR}/hotpotqa_supporting_only_question_features_meta_ok.jsonl"],
    "twowiki_supporting_only_question_features_meta.jsonl": [f"{Q_DIR}/twowiki_supporting_only_question_features_meta_ok.jsonl"],
}

SINGLES = ("attention", "hidden", "probability", "residual")
PAIRS = ("hid_attn", "hid_prob", "hid_resid", "prob_attn", "prob_resid", "resid_attn")
TRIPLES = ("hid_prob_attn", "hid_prob_resid", "hid_resid_attn", "prob_resid_attn")

TEMPLATES = []
for fam, single_dir, pair_dir, triple_dir, sfx in (
        ("q", "L17rd_attn", "L17rd_attn_pairs", "L17rd_attn_triples", "_L17rd_attn"),
        ("ctx", "ctx_L17rd_attn", "ctx_L17rd_attn_pairs", "ctx_L17rd_attn_triples", "_ctxL17rd_attn")):
    for s in SINGLES:
        TEMPLATES.append((fam, f"only_{s}", PROBE / single_dir / f"train_only_{s}{sfx}.yaml"))
    for p in PAIRS:
        TEMPLATES.append((fam, f"pair_{p}", PROBE / pair_dir / f"train_pair_{p}.yaml"))
    for t in TRIPLES:
        TEMPLATES.append((fam, f"triple_{t}", PROBE / triple_dir / f"train_triple_{t}.yaml"))
    TEMPLATES.append((fam, "quad", PROBE / triple_dir / "train_triple_quad.yaml"))


def remap_metas(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        repl = META_MAP.get(Path(p).name)
        if repl is None:
            raise SystemExit(f"no gemma mapping for meta {p}")
        out.extend(repl)
    return out


def main():
    units = []
    for fam, branch, tpl in TEMPLATES:
        cfg = yaml.safe_load(tpl.open())
        cfg["experiment_name"] = f"gemma4_{cfg['experiment_name']}"
        cfg["model_name"] = MODEL
        cfg.pop("gpu_id", None)
        # gemma training-budget amendment (2026-07-17, trajectory evidence): the qwen
        # epoch caps cut off slow-learning resid/attn units (ctx 8/15, q 6/15 still
        # improving at cap; collapsed q seeds need ~100 epochs to escape). Double the
        # caps, patience 16->24 (max observed new-best gap was 12), lr patience 8.
        p = cfg["probe"]
        p["epochs"] = 160 if fam == "q" else 80
        p["early_stopping_patience"] = 24
        p["lr_scheduler_patience"] = 8
        d = cfg["data"]
        d["train_meta_paths"] = remap_metas(d["train_meta_paths"])
        d["validation_meta_paths"] = remap_metas(d["validation_meta_paths"])
        d.pop("judge_consensus_purified", None)
        d.pop("purification_summary_path", None)
        for part in ("train_partition", "validation_partition"):
            if part in d:
                d[part]["ids_path"] = "configs/sampling/gemma4_scope_ids.json"
                d[part]["name"] = d[part]["name"].replace("fixed_", "gemma4_")
        o = cfg["output"]
        for k in ("checkpoint_path", "val_prediction_path"):
            o[k] = o[k].replace("outputs/", "outputs/gemma4/", 1)

        unit = f"{fam}_{branch}"
        path = OUT / fam / f"{unit}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(cfg, path.open("w"), sort_keys=False)
        units.append({"unit": unit, "family": fam, "config": str(path.relative_to(ROOT)),
                      "checkpoint": o["checkpoint_path"], "val_preds": o["val_prediction_path"]})

    # ctx trainings are heavier (64k rows incl rewritten aug) — front-load them
    units.sort(key=lambda u: (u["family"] != "ctx", u["unit"]))
    (OUT / "manifest.json").write_text(json.dumps({"units": units}, indent=1))
    print(f"wrote {len(units)} encoder configs -> {OUT}")


if __name__ == "__main__":
    main()
