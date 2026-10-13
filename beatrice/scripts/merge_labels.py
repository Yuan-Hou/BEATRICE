import argparse

from ragprobe.utils.io import read_jsonl, write_jsonl
from ragprobe.utils.logging import log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-meta", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = read_jsonl(args.labels)
    label_by_question = {
        item["question"]: {
            "emb_entities_raw": item.get("emb_entities_raw", []),
            "attn_entities_raw": item.get("attn_entities_raw", []),
            "emb_target_tokens": item.get("emb_target_tokens", []),
            "attn_target_tokens": item.get("attn_target_tokens", []),
        }
        for item in labels
    }

    records = read_jsonl(args.feature_meta)
    matched = 0
    merged = []

    for item in records:
        q = item.get("example", {}).get("question") or item.get("question")
        if q in label_by_question:
            item.update(label_by_question[q])
            matched += 1
        merged.append(item)

    write_jsonl(merged, args.output)
    log(f"Matched {matched}/{len(merged)} records. Saved to {args.output}")


if __name__ == "__main__":
    main()
