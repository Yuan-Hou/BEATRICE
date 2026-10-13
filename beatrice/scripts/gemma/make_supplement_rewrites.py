"""Context-rewrite pass for supplement candidates that passed base unanimity.

Replicates the original rewritten-context augmentation exactly:
  - rewrite model Qwen/Qwen3.5-27B @2334, temperature 0 (same as
    configs/experiment/analysis/embedding_rewrite_viz_full.yaml)
  - prompt = _context_rewrite_prompt from ragprobe.analysis.embedding_viz
    (copied verbatim below — embedding_viz pulls heavy viz deps)
  - rewrite input = "\n\n".join(f"{title}\n{text}") over example contexts
  - output example format mirrors run_rewritten_context_eval._make_example
    (single context titled "rewritten_context", same eid as base)

Resumable: ids already present in the output file are skipped.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ragprobe.llm.openai_client import build_llm  # noqa: E402

GEN_DIR = ROOT / "data" / "generations"
OUT_PATH = ROOT / "data" / "gemma4" / "sources_supplement" / "rewritten_supplement.jsonl"
NAMES = ("hotpotqa_supporting_only", "musique", "twowiki_supporting_only", "popqa")

REWRITE_MODEL = {
    "name": "Qwen/Qwen3.5-27B",
    "base_url": "http://localhost:2334/v1",
    "api_key": "EMPTY",
    "timeout": 600,
    "max_total_tokens": 16000,
    "context_token_buffer": 256,
    "temperature": 0.0,
    "enable_thinking": False,
    "max_retries": 3,
    "retry_sleep": 1.0,
}


def _context_rewrite_prompt(question: str, context: str) -> str:
    return f"""Rewrite the context into a compact evidence-style passage aligned to the question.

Rules:
- Use only information from the context.
- Preserve named entities, dates, numbers, and relations that may help answer the question.
- Remove irrelevant wording and formatting noise.
- Do not answer the question directly unless the context states the answer.
- Output exactly this format:
Normalized Context: <compact rewritten context>

Question: {question}

Context:
{context}"""


def _strip_prefix(text: str, prefixes: tuple[str, ...]) -> str:
    text = (text or "").strip()
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].strip()
    return text


def unanimous_both_runs(rec: dict) -> bool:
    for run in (rec.get("runs") or {}).values():
        votes = [d.get("parsed") for d in (run.get("judge_vote") or {}).get("variants", [])]
        if len(votes) != 3 or any(v is None for v in votes) or len(set(votes)) != 1:
            return False
    return True


def context_text(example: dict) -> str:
    parts = []
    for ctx in example.get("contexts") or []:
        title = str(ctx.get("title") or "").strip()
        text = str(ctx.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"{title}\n{text}".strip() if title else text)
    return "\n\n".join(parts).strip()


def main() -> None:
    pending = []
    for name in NAMES:
        p = GEN_DIR / f"supp_{name}_gemma4e4b_results.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            rec = json.loads(line)
            if unanimous_both_runs(rec):
                pending.append(rec["example"])

    done: set[str] = set()
    if OUT_PATH.exists():
        done = {json.loads(l)["id"] for l in OUT_PATH.open()}
    todo = [ex for ex in pending if ex["id"] not in done]
    print(f"base-unanimous candidates: {len(pending)}, already rewritten: {len(done)}, todo: {len(todo)}")
    if not todo:
        return

    llm = build_llm(REWRITE_MODEL)

    def rewrite(ex: dict) -> dict | None:
        ctx = context_text(ex)
        if not ctx:
            return None
        raw = llm.generate(_context_rewrite_prompt(ex["question"], ctx))
        rewritten = _strip_prefix(raw, ("Normalized Context:",))
        if not rewritten:
            return None
        meta = dict(ex.get("metadata") or {})
        meta.update({
            "rewrite_status": "success",
            "rewrite_model": REWRITE_MODEL["name"],
            "original_context_chars": len(ctx),
            "rewritten_context_chars": len(rewritten),
            "supplement": True,
        })
        return {
            "id": ex["id"],
            "dataset": ex.get("dataset"),
            "split": ex.get("split"),
            "question": ex["question"],
            "contexts": [{
                "title": "rewritten_context",
                "text": rewritten,
                "source": "embedding_rewrite_viz",
                "is_supporting": True,
                "metadata": {"context_count_original": len(ex.get("contexts") or [])},
            }],
            "answers": ex.get("answers") or [],
            "answerable": ex.get("answerable", True),
            "metadata": meta,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with OUT_PATH.open("a") as f, ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(rewrite, ex): ex["id"] for ex in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="rewrite"):
            try:
                rec = fut.result()
            except Exception as exc:
                print(f"rewrite failed {futures[fut]}: {exc}")
                continue
            if rec is not None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
    print(f"rewrote {n_ok}/{len(todo)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
