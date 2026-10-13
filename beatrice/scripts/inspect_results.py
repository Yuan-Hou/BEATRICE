import argparse
from collections import Counter

from ragprobe.utils.io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    records = read_jsonl(args.path)
    print(f"Records: {len(records)}")

    transitions = Counter(r.get("derived", {}).get("transition", "??") for r in records)
    for k in ["NY", "YN", "YY", "NN", "??"]:
        count = transitions.get(k, 0)
        ratio = count / max(len(records), 1)
        print(f"{k}: {count} ({ratio:.2%})")


if __name__ == "__main__":
    main()
