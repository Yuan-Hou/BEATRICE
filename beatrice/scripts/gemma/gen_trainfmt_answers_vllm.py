"""Batch-generate TRAINING-FORMAT answers for gemma feature extraction via vLLM.

Rationale (user 2026-07-16): while the -it vLLM service is up, pre-generate the
greedy answers the probe features are computed on, so the HF-side extractor
only ever does teacher-forced forwards (answer_override), never generation.

Prompt parity: prompts are built with the SAME functions the extractor uses
(_build_prompt = QUESTION_ONLY / LEGACY_MUSIQUE) and sent through the chat API,
so the server applies the same canonical chat_template.jinja the HF path uses
(enable_thinking off, temperature 0, max_tokens = answer_tokens cap).

Output: data/gemma4/trainfmt_answers/<name>.jsonl, one row per example:
  {"id": ..., "dataset": ..., "answers": {"with_context": ..., "without_context": ...}}
(rewritten_context: with_context only.)

Usage:
  python scripts/gemma/gen_trainfmt_answers_vllm.py \
      --sources-dir data/gemma4/sources_final [--names musique,...] \
      [--limit N] [--workers 32]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI  # noqa: E402
from tqdm import tqdm  # noqa: E402

from ragprobe.features.extractor import _build_prompt, _format_context_legacy  # noqa: E402

MODEL = "google/gemma-4-E4B-it"
BASE_URL = "http://localhost:2335/v1"
ANSWER_TOKENS = 32
NAMES = ("musique", "hotpotqa_supporting_only", "twowiki_supporting_only",
         "popqa", "nq", "triviaqa", "rewritten_context")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources-dir", default="data/gemma4/sources_final")
    ap.add_argument("--names", default=",".join(NAMES))
    ap.add_argument("--out-dir", default="data/gemma4/trainfmt_answers")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key="EMPTY", timeout=180)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def ask(prompt: str) -> str:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=ANSWER_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (r.choices[0].message.content or "").strip()

    for name in args.names.split(","):
        src = ROOT / args.sources_dir / f"{name}.jsonl"
        out = out_dir / f"{name}.jsonl"
        rows = [json.loads(l) for l in src.open()]
        if args.limit:
            rows = rows[:args.limit]
        done = {}
        if out.exists():  # resume by id
            for l in out.open():
                r = json.loads(l)
                done[r["id"]] = r
        todo = [ex for ex in rows if ex["id"] not in done]
        settings = ("with_context",) if name == "rewritten_context" else ("with_context", "without_context")
        print(f"[{name}] {len(todo)}/{len(rows)} to generate ({'+'.join(settings)})", flush=True)

        def work(ex):
            q = str(ex.get("question") or "").strip()
            ans = {}
            for s in settings:
                ctx = _format_context_legacy({"example": ex}).strip() if s == "with_context" else None
                _, prompt = _build_prompt(q, ctx)
                ans[s] = ask(prompt)
            return {"id": ex["id"], "dataset": ex.get("dataset"), "answers": ans}

        with out.open("a") as fh, ThreadPoolExecutor(args.workers) as pool:
            for i, r in enumerate(tqdm(pool.map(work, todo), total=len(todo), desc=name)):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                if (i + 1) % 500 == 0:
                    fh.flush()
        n = sum(1 for _ in out.open())
        print(f"[{name}] done: {n}/{len(rows)} rows in {out}", flush=True)
    print("TRAINFMT_ANSWERS_DONE", flush=True)


if __name__ == "__main__":
    main()
