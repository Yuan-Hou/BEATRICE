#!/usr/bin/env python
"""Token-span ablation: merge caches + eval_subset + report.

Rows (encoders RETRAINED with the segment's features zeroed at train time, then the
exact eval_subset protocol: real fusion head class, 3 seeds, val-AUROC selection):

  with-ctx (joint champion  = 2 ctx branches + q_hid_attn closed-book):
    full / drop context / drop question / drop answer
    (drop-context leaves the closed-book branch untouched — it never sees context;
     drop-question/answer also retrain the closed-book branch without that segment)
  q-only (champion = hid_resid + hid_prob_attn + hid_attn):
    full / drop question / drop answer

Reads outputs/cached_fusion_vecs/{joint_30br,qonly_all15,span_ablation/*}
Writes outputs/reports/span_ablation.{json,md}

Usage: CUDA_VISIBLE_DEVICES=<gpu> python scripts/span_ablation_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_fusion_combos import eval_subset, load_split  # noqa: E402

R = Path("outputs/reports")
CV = Path("outputs/cached_fusion_vecs")
SA = CV / "span_ablation"
CTX_CHAMP = ["hid_prob_attn", "hid_prob_resid", "q_hid_attn"]
QONLY_CHAMP = ["hid_resid", "hid_prob_attn", "hid_attn"]
EVAL_SPLITS = ["test", "popqa", "nq", "triviaqa"]
ALL_SPLITS = ["train", "validation"] + EVAL_SPLITS
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def merged_ctx_split(cond_cache: Path, q_cache: Path, split: str, q_branch="hid_attn"):
    """One with-ctx split with the closed-book branch joined in (eid-aligned,
    right-zero-padded to the ctx vec dim — same convention as joint_30br)."""
    c = torch.load(cond_cache / f"{split}.pt", map_location="cpu", weights_only=False)
    q = torch.load(q_cache / f"{split}.pt", map_location="cpu", weights_only=False)
    qmap = {e: i for i, e in enumerate(q["example_ids"])}
    idx = torch.tensor([qmap[e] for e in c["example_ids"]], dtype=torch.long)
    qv = q["vecs"][q_branch][idx]
    dim = next(iter(c["vecs"].values())).shape[1]
    if qv.shape[1] < dim:
        qv = torch.nn.functional.pad(qv, (0, dim - qv.shape[1]))
    vecs = {**c["vecs"], f"q_{q_branch}": qv}
    logits = {**c["logits"], f"q_{q_branch}": q["logits"][q_branch][idx]}
    return {"vecs": {k: v.to(DEV) for k, v in vecs.items()},
            "logits": {k: v.to(DEV) for k, v in logits.items()},
            "y": c["labels"].reshape(-1).to(device=DEV, dtype=torch.float32)}


def main():
    seeds = [0, 1, 2]
    conditions = []  # (setting, name, splits_loader, side, subset)

    # ---- with-ctx: full + three span drops ----
    conditions.append(("withctx", "full",
                       lambda: {s: load_split(CV / "joint_30br", s, DEV) for s in ALL_SPLITS},
                       "context", CTX_CHAMP))
    for name, q_cache in (("drop_context", CV / "qonly_all15"),
                          ("drop_question", SA / "qonly_dropq"),
                          ("drop_answer", SA / "qonly_dropa")):
        cond_cache = SA / {"drop_context": "ctx_dropc", "drop_question": "ctx_dropq",
                           "drop_answer": "ctx_dropa"}[name]
        conditions.append(("withctx", name,
                           (lambda cc=cond_cache, qc=q_cache:
                            {s: merged_ctx_split(cc, qc, s) for s in ALL_SPLITS}),
                           "context", CTX_CHAMP))

    # ---- q-only: full + two span drops ----
    for name, cache in (("full", CV / "qonly_all15"),
                        ("drop_question", SA / "qonly_dropq"),
                        ("drop_answer", SA / "qonly_dropa")):
        conditions.append(("qonly", name,
                           (lambda c=cache: {s: load_split(c, s, DEV) for s in ALL_SPLITS}),
                           "question", QONLY_CHAMP))

    out = {"protocol": "encoders retrained with segment zeroed; eval_subset, 3-seed mean",
           "withctx": {}, "qonly": {}}
    for setting, name, loader, side, subset in conditions:
        splits = loader()
        r = eval_subset(subset, splits, side, DEV, seeds, eval_splits=EVAL_SPLITS)
        r["branches"] = subset
        out[setting][name] = r
        print(f"[{setting}/{name}] val={r['val']:.4f} " +
              " ".join(f"{s}={r[s]:.4f}" for s in EVAL_SPLITS), flush=True)
        del splits

    json.dump(out, open(R / "span_ablation.json", "w"), indent=1)

    lines = ["# Token-span ablation (drop one of question/answer/context; encoders "
             "retrained with the segment's features zeroed; eval_subset 3-seed protocol)",
             "", "AUROC top / AUPRC (LLM answered WRONG = positive) bottom.", ""]
    for setting, order in (("withctx", ["full", "drop_context", "drop_question", "drop_answer"]),
                           ("qonly", ["full", "drop_question", "drop_answer"])):
        lines += [f"## {setting}", "",
                  "| condition | val | " + " | ".join(EVAL_SPLITS) + " |",
                  "|---|---|" + "---|" * len(EVAL_SPLITS)]
        for name in order:
            r = out[setting].get(name)
            if not r:
                continue
            cells = [f"{r['val']:.4f}<br>{r.get('val_auprc', float('nan')):.4f}"]
            cells += [f"{r[s]:.4f}<br>{r.get(f'{s}_auprc', float('nan')):.4f}"
                      for s in EVAL_SPLITS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    (R / "span_ablation.md").write_text("\n".join(lines) + "\n")
    print("SPAN_ABLATION_EVAL_DONE")


if __name__ == "__main__":
    main()
