"""Gold-standard smoke test for the probe scorer service.

Takes N validation examples, feeds their (question, contexts, training-time answer)
through the ONLINE ProbeScorer, and compares p_correct against the OFFLINE eval
predictions for the same example_ids (outputs/predictions/..._validation_..._preds.jsonl).
If the online feature extraction + probe forward match the training/eval pipeline, the
probabilities agree (small tolerance for answer re-tokenization).

    CUDA_VISIBLE_DEVICES=<free> python scripts/smoke_test_probe_scorer.py \
        --config configs/experiment/services/serve_probe_scorer_qwen9b.yaml -n 5
"""
from __future__ import annotations

import argparse
import json

import torch

from ragprobe.services.probe_scorer import ProbeScorer
from ragprobe.utils.config import load_yaml

OFFLINE_PREDS = "outputs/predictions/frozen_fusion_context_validation_ctxL17rd_attn_preds.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    offline = {}
    for line in open(OFFLINE_PREDS):
        r = json.loads(line)
        offline[r["example_id"]] = r

    scorer = ProbeScorer(cfg)

    print(f"\ncomparing {args.n} validation examples (online vs offline eval):")
    diffs = []
    tested = 0
    for r in offline.values():
        src = r.get("source_meta") or {}
        ex = src.get("example") or {}
        sidecar_answer = (src.get("feature_extraction", {}).get("rag", {})
                          .get("spans", {}).get("answer_text"))
        if not ex.get("question") or not ex.get("contexts") or not sidecar_answer:
            continue
        res = scorer.score({
            "example_id": r["example_id"],
            "question": ex["question"],
            "contexts": ex["contexts"],
            "answer": sidecar_answer,
        })
        if res.get("status") != "success":
            print(f"  {r['example_id']}: FAILED {res.get('error')}")
            continue
        d = abs(res["p_correct"] - r["binary_probability"])
        diffs.append(d)
        print(f"  {r['example_id']}: online={res['p_correct']:.4f} "
              f"offline={r['binary_probability']:.4f} |Δ|={d:.4f} "
              f"(label={r.get('label')}, answer={res['answer'][:40]!r})")
        tested += 1
        if tested >= args.n:
            break

    assert tested > 0, "no comparable examples found"
    mx = max(diffs)
    print(f"\nmax |Δp| = {mx:.4f} over {tested} examples")
    if mx < 0.02:
        print("PROBE SCORER SMOKE TEST PASSED")
    else:
        print("PROBE SCORER MISMATCH — investigate before serving")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
