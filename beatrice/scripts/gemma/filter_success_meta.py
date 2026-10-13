"""Write the success-only view of an extraction meta (gemma pipeline).

gemma-4-E4B occasionally greedy-generates an empty answer (immediate
<end_of_turn>); the extractor records those rows as failed. Unlike Qwen (0
failures in 11k), these are expected — but ONLY that reason, and only at a low
rate. Anything else means a real pipeline problem, so we hard-fail.

Output: <meta stem>_ok.jsonl next to the input; prints FILTER_META_OK on success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--max-fail-rate", type=float, default=0.05)
    args = ap.parse_args()

    meta = Path(args.meta)
    rows = [json.loads(l) for l in meta.open()]
    ok, bad_reasons, dropped = [], {}, []
    for r in rows:
        fe = r.get("feature_extraction", {})
        if fe.get("status") == "success":
            ok.append(r)
        else:
            reason = str(fe.get("reason"))
            dropped.append({"id": r.get("example", {}).get("id"), "reason": reason})
            if reason not in ("empty_generated_answer", "missing_record_answer"):
                bad_reasons[reason] = bad_reasons.get(reason, 0) + 1

    rate = (len(rows) - len(ok)) / max(len(rows), 1)
    print(f"{meta.name}: {len(ok)}/{len(rows)} success, fail rate {rate:.2%}")
    if bad_reasons:
        print(f"FILTER_META_FAILED unexpected reasons: {bad_reasons}")
        raise SystemExit(1)
    if rate > args.max_fail_rate:
        print(f"FILTER_META_FAILED fail rate {rate:.2%} > {args.max_fail_rate:.2%}")
        raise SystemExit(1)

    out = meta.with_name(meta.stem + "_ok.jsonl")
    with out.open("w") as fh:
        for r in ok:
            fh.write(json.dumps(r) + "\n")
    if dropped:
        drop_path = meta.with_name(meta.stem + "_dropped.json")
        drop_path.write_text(json.dumps(dropped, indent=1))
    print(f"FILTER_META_OK -> {out} (dropped {len(dropped)})")


if __name__ == "__main__":
    main()
