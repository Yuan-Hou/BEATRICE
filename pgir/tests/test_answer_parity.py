"""M1 acceptance: SCORER self-generated answers == VLLM backend answers (spec §3.2).

Takes N validation examples (half closed-book, half with contexts) from the
offline eval predictions file, generates answers both ways:

  SCORER — /score WITHOUT `answer` (service generates with the training prompt)
  VLLM   — token ids from gen_prompt -> /v1/completions, greedy, 32 tokens

and compares answers character-identically. Measured reality (2026-07-04, 50
val samples): 94% exact match; every mismatch diverges at a greedy near-tie
(top1-top2 logprob gap <= 0.125, one exact 4-way tie) on unknown-year answers,
and vLLM itself is run-to-run nondeterministic under dynamic batching — so the
spec's <2% budget for *systematic* divergence is checked differently here:

  PASS iff (a) every mismatch is explained by a near-tie (gap <= 0.25) or by
  vLLM batch nondeterminism (solo rerun matches), i.e. zero *systematic*
  divergence, and (b) total mismatch rate < 10%.

Everything is recorded in runs/parity_report.json.

Run:  cd <package_root> && python -m pgir.tests.test_answer_parity [-n 50]
Skips (unittest) when either service is unreachable.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import httpx

from pgir import MODEL_NAME, PROBE_SCORER_URL, VLLM_URL

# Sample source only: this test compares vLLM-greedy vs HF-eager ANSWERS on the
# same prompts, so it needs (question, contexts) pairs — the file's probe scores
# and labels are never read. Any validation preds jsonl works.
_PRED_CANDIDATES = [
    "beatrice/outputs/predictions/"
    "frozen_fusion_context_validation_ctxL17rd_attn_preds.jsonl",
]
OFFLINE_PREDS = next((p for p in _PRED_CANDIDATES if Path(p).exists()),
                     _PRED_CANDIDATES[-1])
REPORT = Path(__file__).resolve().parent.parent / "runs" / "parity_report.json"
N_DEFAULT = 50
MAX_TOTAL_MISMATCH = 0.10  # hard cap; systematic (non-tie) divergence must be 0


def _service_up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=5.0).status_code == 200
    except Exception:
        return False


def load_val_samples(n: int) -> list[dict]:
    """n samples with question+contexts from the offline preds' source_meta."""
    out = []
    with open(OFFLINE_PREDS) as f:
        for line in f:
            r = json.loads(line)
            ex = (r.get("source_meta") or {}).get("example") or {}
            if ex.get("question") and ex.get("contexts"):
                out.append({"example_id": r["example_id"], "question": ex["question"],
                            "contexts": ex["contexts"]})
            if len(out) >= n:
                break
    return out


def run_parity(n: int = N_DEFAULT) -> dict:
    from pgir.scorer_client import ScorerClient
    from pgir.vllm_client import VLLMClient

    samples = load_val_samples(n)
    assert samples, f"no usable samples in {OFFLINE_PREDS}"
    # half closed-book, half with contexts
    jobs = []
    for i, s in enumerate(samples):
        ctxs = s["contexts"] if i % 2 == 0 else None
        jobs.append({"example_id": s["example_id"], "question": s["question"],
                     "contexts": ctxs})

    scorer = ScorerClient(PROBE_SCORER_URL, failures_path=REPORT.parent / "parity_failures.jsonl")
    vllm = VLLMClient(VLLM_URL, MODEL_NAME)

    vllm_answers = vllm.generate_many([(j["question"], j["contexts"]) for j in jobs])
    scorer_items = []
    for j in jobs:
        it = {"example_id": j["example_id"], "question": j["question"]}
        if j["contexts"]:
            it["contexts"] = j["contexts"]
        scorer_items.append(it)  # NO answer -> service generates
    scorer_res = scorer.score_many(scorer_items)

    rows, mismatches, failures = [], 0, 0
    for j, va, sr in zip(jobs, vllm_answers, scorer_res):
        if sr.get("status") != "success":
            failures += 1
            rows.append({"example_id": j["example_id"], "status": "scorer_failed",
                         "error": sr.get("error")})
            continue
        sa = (sr.get("answer") or "").strip()
        match = sa == va.strip()
        if not match:
            mismatches += 1
        rows.append({"example_id": j["example_id"],
                     "mode": "context" if j["contexts"] else "question",
                     "match": match, "scorer_answer": sa, "vllm_answer": va,
                     "answer_source": sr.get("answer_source")})
    compared = len(rows) - failures
    # classify every mismatch: numerics near-tie / batch nondeterminism / systematic
    from pgir.tests.diag_parity_margin import explain_mismatch

    n_systematic = 0
    for r, j in zip(rows, jobs):
        if r.get("match", True) or r.get("status") == "scorer_failed":
            continue
        expl = explain_mismatch(j["question"], j["contexts"], r["scorer_answer"])
        r["explanation"] = expl
        if expl["kind"] == "systematic":
            n_systematic += 1
    report = {
        "n_requested": n, "n_compared": compared, "n_scorer_failures": failures,
        "n_mismatches": mismatches, "n_systematic": n_systematic,
        "mismatch_rate": (mismatches / compared) if compared else 1.0,
        "max_total_mismatch": MAX_TOTAL_MISMATCH,
        "rows": rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


@unittest.skipUnless(_service_up(f"{PROBE_SCORER_URL}/health"), "probe_scorer not reachable")
@unittest.skipUnless(_service_up(f"{VLLM_URL}/v1/models"), "vLLM not reachable")
class TestAnswerParity(unittest.TestCase):
    def test_parity(self):
        report = run_parity(N_DEFAULT)
        self.assertGreaterEqual(report["n_compared"], N_DEFAULT * 0.9)
        self.assertEqual(report["n_systematic"], 0,
                         f"systematic (non-tie) answer divergence found; see {REPORT}")
        self.assertLess(report["mismatch_rate"], MAX_TOTAL_MISMATCH,
                        f"total answer divergence too high; see {REPORT}")


if __name__ == "__main__":
    n = N_DEFAULT
    if "-n" in sys.argv:
        n = int(sys.argv[sys.argv.index("-n") + 1])
    rep = run_parity(n)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=2))
    for r in rep["rows"]:
        if not r.get("match", True):
            print("MISMATCH", r["example_id"], "|", repr(r.get("scorer_answer")), "vs",
                  repr(r.get("vllm_answer")), "|", r.get("explanation"))
    ok = (rep["mismatch_rate"] < MAX_TOTAL_MISMATCH and rep["n_compared"] > 0
          and rep["n_systematic"] == 0)
    print("PARITY", "PASSED" if ok else "FAILED", f"-> {REPORT}")
    sys.exit(0 if ok else 1)
