"""Encoder-seed replicas for the aligned qonly campaign (user, 2026-07-15).

Motivation: under aligned labels the greedy champion {hid_attn, hid_prob, residual}
no longer strictly dominates hidden-only on test/popqa (deltas -0.0006/-0.0023,
below the ~0.003 encoder-seed noise of the old seedmean protocol). To (a) report
champion rows as encoder-seedmean like the old campaign and (b) re-run the
val-only subset selection under seedmean, we retrain every qonly encoder twice
more (training is unseeded, so each rerun is an independent init) and precompute
a full all15 cache per replica run.

Priority: the 6 encoders touching the val-equivalence-class subsets + the
hidden-only ablation baseline run first, so the seedmean re-selection can start
before the tail replicas finish.

GPUs: launch with SCHED_EXCLUDE_GPUS=0,2,3,4,5,7 (i.e. gpu1+gpu6 only) so the
main campaign driver's pool (gpu0/4/7) is untouched.

Marker: QONLY_SEEDREP_DONE / QONLY_SEEDREP_FAILED in the driver log.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(__file__))
import ctx_auto_scheduler as _cas  # noqa: E402
from ctx_auto_scheduler import Job, schedule  # noqa: E402

_cas.MIN_FREE_GB = float(os.environ.get("SCHED_MIN_FREE_GB", "28"))

ROOT = Path(__file__).resolve().parents[1]
SRC_CFG = ROOT / "configs/experiment/probe/aligned"
DST_CFG = ROOT / "configs/experiment/probe/aligned_seedrep"
LOG = "logs/aligned/seedrep"
RUNS = ("run2", "run3")

# equivalence-class subsets (exhaustive_k3_qonly ranks 1-3) + hidden baseline
KEY_TAGS = ["pair_hid_attn", "pair_hid_prob", "only_residual",
            "triple_hid_prob_attn", "triple_prob_resid_attn", "only_hidden"]
ALL_TAGS = ([f"only_{b}" for b in ("hidden", "probability", "residual", "attention")]
            + [f"pair_{b}" for b in ("hid_attn", "hid_prob", "hid_resid",
                                     "prob_attn", "prob_resid", "resid_attn")]
            + [f"triple_{b}" for b in ("hid_prob_attn", "hid_prob_resid",
                                       "hid_resid_attn", "prob_resid_attn")]
            + ["quad"])
ORDERED = KEY_TAGS + [t for t in ALL_TAGS if t not in KEY_TAGS]

# precompute branch name -> encoder tag (mirrors precompute_question_all15.yaml)
BRANCH_TAG = {"hidden": "only_hidden", "probability": "only_probability",
              "residual": "only_residual", "attention": "only_attention",
              "hid_attn": "pair_hid_attn", "hid_prob": "pair_hid_prob",
              "hid_resid": "pair_hid_resid", "prob_attn": "pair_prob_attn",
              "prob_resid": "pair_prob_resid", "resid_attn": "pair_resid_attn",
              "hid_prob_attn": "triple_hid_prob_attn",
              "hid_prob_resid": "triple_hid_prob_resid",
              "hid_resid_attn": "triple_hid_resid_attn",
              "prob_resid_attn": "triple_prob_resid_attn", "quad": "quad"}


def ckpt(run: str, tag: str) -> str:
    return f"outputs/aligned/probes_seedrep/{run}/question/{tag}/question_checkpoint.pth"


def make_train_cfg(run: str, tag: str) -> str:
    c = yaml.safe_load(open(SRC_CFG / f"train_question_{tag}.yaml"))
    c["experiment_name"] += f"_{run}"
    out = f"outputs/aligned/probes_seedrep/{run}/question/{tag}"
    c["output"]["checkpoint_path"] = f"{out}/question_checkpoint.pth"
    c["output"]["val_prediction_path"] = f"{out}/question_val_preds.jsonl"
    dst = DST_CFG / f"train_question_{tag}_{run}.yaml"
    yaml.safe_dump(c, open(dst, "w"), sort_keys=False, allow_unicode=True)
    return str(dst.relative_to(ROOT))


def make_pc_cfg(run: str) -> str:
    c = yaml.safe_load(open(SRC_CFG / "precompute_question_all15.yaml"))
    c["experiment_name"] += f"_{run}"
    for br in c["probe"]["branch_checkpoints"]:
        br["checkpoint_path"] = ckpt(run, BRANCH_TAG[br["name"]])
    c["output"]["checkpoint_path"] = f"outputs/aligned/checkpoints/_unused_precompute_question_{run}.pth"
    dst = DST_CFG / f"precompute_question_all15_{run}.yaml"
    yaml.safe_dump(c, open(dst, "w"), sort_keys=False, allow_unicode=True)
    return str(dst.relative_to(ROOT))


def main() -> None:
    DST_CFG.mkdir(parents=True, exist_ok=True)
    (ROOT / LOG).mkdir(parents=True, exist_ok=True)
    jobs: dict[str, Job] = {}
    for tag in ORDERED:                      # key tags first; dict order = schedule order
        for run in RUNS:
            name = f"q_{tag}_{run}"
            jobs[name] = Job(name, "train", make_train_cfg(run, tag),
                             f"{LOG}/train_question_{tag}_{run}.log",
                             "Done. Best epoch", ckpt(run, tag))
    for run in RUNS:
        cache = f"outputs/aligned/cached_fusion_vecs/question_all15_{run}"
        extra = ["--splits", "train,validation", "--side", "question"]
        for split in ("test", "popqa", "nq", "triviaqa"):
            extra += ["--eval-config",
                      f"{split}=configs/experiment/probe/aligned/eval_question_{split}.yaml"]
        jobs[f"pc_question_{run}"] = Job(
            f"pc_question_{run}", "precompute",
            make_pc_cfg(run), f"{LOG}/precompute_question_{run}.log",
            "PRECOMPUTE_DONE", f"{cache}/triviaqa.pt",
            deps=[f"q_{t}_{run}" for t in ALL_TAGS],
            cache_dir=cache, extra_args=extra)

    ok = schedule(jobs)
    failed = [j.name for j in jobs.values() if j.status not in ("done", "skipped")]
    if failed or not ok:
        print(f"QONLY_SEEDREP_FAILED: {failed}", flush=True)
        sys.exit(1)
    print("QONLY_SEEDREP_DONE", flush=True)


if __name__ == "__main__":
    main()
