"""Diagnose parity mismatches: how close was the greedy race at each token?

For a mismatched sample, re-run vLLM generation with logprobs=5 and report the
top-1 vs top-2 logprob gap at every step, marking where the SCORER answer's
token disagrees with vLLM's pick and how far behind it was. A tiny gap at the
divergence point == numerics-driven near-tie, not a prompt/decoding bug.

Usage: cd <package_root> && python -m pgir.tests.diag_parity_margin <example_id> ...
       (default: all mismatches in runs/parity_report.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

from pgir import MODEL_NAME, VLLM_URL
from pgir.gen_prompt import MAX_ANSWER_TOKENS, load_processor, prompt_token_ids
from pgir.tests.test_answer_parity import OFFLINE_PREDS, REPORT


NEAR_TIE_GAP = 0.25


def explain_mismatch(question: str, contexts, scorer_answer: str) -> dict:
    """Solo-rerun vLLM with logprobs and classify one parity mismatch.

    Returns {"kind": "batch_nondeterminism" | "near_tie" | "systematic",
             "gap": float | None, "rerun_answer": str}.
    """
    proc = load_processor(MODEL_NAME)
    tok = proc.tokenizer
    ids = prompt_token_ids(proc, question, contexts)
    r = httpx.post(f"{VLLM_URL}/v1/completions", timeout=180.0, json={
        "model": MODEL_NAME, "prompt": ids, "max_tokens": MAX_ANSWER_TOKENS,
        "temperature": 0.0, "logprobs": 5,
    })
    r.raise_for_status()
    ch = r.json()["choices"][0]
    rerun = ch["text"].strip()
    if rerun == scorer_answer.strip():
        return {"kind": "batch_nondeterminism", "gap": None, "rerun_answer": rerun}
    scorer_ids = tok(scorer_answer, add_special_tokens=False)["input_ids"]
    lp = ch["logprobs"]
    for i, (t, top) in enumerate(zip(lp["tokens"], lp["top_logprobs"])):
        scorer_tok = tok.decode([scorer_ids[i]]) if i < len(scorer_ids) else None
        if scorer_tok is None or scorer_tok == t:
            continue
        srt = sorted(top.items(), key=lambda kv: -kv[1])
        scorer_lp = dict(srt).get(scorer_tok)
        gap = (srt[0][1] - scorer_lp) if scorer_lp is not None else float("inf")
        kind = "near_tie" if gap <= NEAR_TIE_GAP else "systematic"
        return {"kind": kind, "gap": gap, "rerun_answer": rerun, "diverge_pos": i}
    # answers differ only in length/whitespace
    return {"kind": "near_tie", "gap": 0.0, "rerun_answer": rerun}


def sample_by_id(eid: str) -> dict:
    with open(OFFLINE_PREDS) as f:
        for line in f:
            r = json.loads(line)
            if r["example_id"] == eid:
                ex = r["source_meta"]["example"]
                return {"question": ex["question"], "contexts": ex["contexts"]}
    raise KeyError(eid)


def main():
    report = json.loads(Path(REPORT).read_text())
    wanted = sys.argv[1:] or [r["example_id"] for r in report["rows"] if not r.get("match", True)]
    rows = {r["example_id"]: r for r in report["rows"]}
    proc = load_processor(MODEL_NAME)
    tok = proc.tokenizer

    for eid in wanted:
        row = rows[eid]
        s = sample_by_id(eid)
        ctxs = s["contexts"] if row.get("mode") == "context" else None
        ids = prompt_token_ids(proc, s["question"], ctxs)
        r = httpx.post(f"{VLLM_URL}/v1/completions", timeout=180.0, json={
            "model": MODEL_NAME, "prompt": ids, "max_tokens": MAX_ANSWER_TOKENS,
            "temperature": 0.0, "logprobs": 5,
        })
        r.raise_for_status()
        ch = r.json()["choices"][0]
        scorer_ids = tok(row["scorer_answer"], add_special_tokens=False)["input_ids"]
        lp = ch["logprobs"]
        print(f"\n=== {eid} (prompt {len(ids)} tokens) ===")
        print(f"scorer: {row['scorer_answer']!r}\nvllm  : {row['vllm_answer']!r}")
        for i, (t, top) in enumerate(zip(lp["tokens"], lp["top_logprobs"])):
            srt = sorted(top.items(), key=lambda kv: -kv[1])
            gap = srt[0][1] - srt[1][1] if len(srt) > 1 else float("inf")
            scorer_tok = tok.decode([scorer_ids[i]]) if i < len(scorer_ids) else None
            diverged = scorer_tok is not None and scorer_tok != t
            mark = "  <-- DIVERGE" if diverged else ""
            print(f"  [{i:02d}] vllm={t!r:20} gap(top1-top2)={gap:7.4f} "
                  f"scorer={scorer_tok!r}{mark}")
            if diverged:
                in_top = {k: round(v, 4) for k, v in srt}
                print(f"        top5: {in_top}")
                break


if __name__ == "__main__":
    main()
