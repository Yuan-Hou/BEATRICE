from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ragprobe.evaluation.answer_extraction import extract_final_answer
from ragprobe.evaluation.generation_runner import run_judge
from ragprobe.evaluation.metrics import add_derived_fields
from ragprobe.llm.openai_client import build_llm
from ragprobe.prompts.qa import build_qa_prompt
from ragprobe.utils.config import load_yaml
from ragprobe.utils.io import atomic_write_jsonl, read_jsonl
from ragprobe.utils.scope import load_scope_ids, scope_path_from_config
from ragprobe.utils.logging import log


def _result_key(record: dict[str, Any]) -> str:
    return str(record.get("example_id") or record.get("id") or record.get("question") or "")


def _load_source_cache(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    paths = sorted({str(r.get("source_path") or "") for r in rows if r.get("source_path")})
    return {path: read_jsonl(path) for path in paths}


def _source_record(row: dict[str, Any], cache: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    path = str(row.get("source_path") or "")
    idx = row.get("source_record_index")
    if path and idx is not None and path in cache:
        records = cache[path]
        idx = int(idx)
        if 0 <= idx < len(records):
            return records[idx]
    return {}


def _transition(no_correct: bool, rag_correct: bool) -> str:
    return ("Y" if no_correct else "N") + ("Y" if rag_correct else "N")


def _make_example(row: dict[str, Any], source: dict[str, Any], question_field: str) -> dict[str, Any]:
    source_example = source.get("example") or {}
    question = str(row.get(question_field) or row.get("question") or source_example.get("question") or "").strip()
    rewritten_context = str(row.get("rewritten_context") or row.get("context") or "").strip()
    answers = row.get("answers") or source_example.get("answers") or []
    if not isinstance(answers, list):
        answers = [answers]
    metadata = dict(source_example.get("metadata") or {})
    metadata.update({
        "source_path": row.get("source_path"),
        "source_record_index": row.get("source_record_index"),
        "rewrite_status": row.get("rewrite_status"),
        "original_question": row.get("question"),
        "rewritten_question": row.get("rewritten_question"),
        "original_context_chars": len(str(row.get("context") or "")),
        "rewritten_context_chars": len(rewritten_context),
    })
    return {
        "id": str(row.get("example_id") or source_example.get("id") or question),
        "dataset": row.get("dataset") or source_example.get("dataset"),
        "split": row.get("split") or source_example.get("split"),
        "question": question,
        "contexts": [{
            "title": "rewritten_context",
            "text": rewritten_context,
            "source": "embedding_rewrite_viz",
            "is_supporting": True,
            "metadata": {"context_count_original": row.get("context_count")},
        }],
        "answers": answers,
        "answerable": source_example.get("answerable", True),
        "metadata": metadata,
    }


def _evaluate_one(row: dict[str, Any], source: dict[str, Any], model_llm, judge_llm, config: dict[str, Any]) -> dict[str, Any]:
    question_field = str(config.get("input", {}).get("question_field", "question"))
    example = _make_example(row, source, question_field)
    prompt = build_qa_prompt(example, with_context=True)
    raw_answer = model_llm.generate(prompt)
    extracted_answer = extract_final_answer(raw_answer)
    judge_result = run_judge(judge_llm, example, extracted_answer, config)

    source_runs = source.get("runs") or {}
    without_context = dict(source_runs.get("without_context") or {})
    if "is_correct" not in without_context:
        without_context["is_correct"] = False

    result = {
        "example": example,
        "model": {
            "name": config["model"]["name"],
            "backend": config["model"].get("backend", "openai_compatible"),
            "base_url": config["model"].get("base_url"),
        },
        "runs": {
            "without_context": without_context,
            "with_context": {
                "prompt": prompt,
                "raw_answer": raw_answer,
                "extracted_answer": extracted_answer,
                **judge_result,
            },
        },
        "rewrite": {
            "source_path": row.get("source_path"),
            "source_record_index": row.get("source_record_index"),
            "rewrite_status": row.get("rewrite_status"),
            "question_field": question_field,
            "original_question": row.get("question"),
            "rewritten_question": row.get("rewritten_question"),
            "rewritten_context": row.get("rewritten_context"),
            "original_transition": (source.get("derived") or {}).get("transition"),
        },
    }
    result = add_derived_fields(result)
    no_correct = bool(result["runs"]["without_context"].get("is_correct"))
    rag_correct = bool(result["runs"]["with_context"].get("is_correct"))
    result.setdefault("derived", {})["transition"] = _transition(no_correct, rag_correct)
    result["derived"]["rewritten_context_transition"] = result["derived"]["transition"]
    result["derived"]["original_transition"] = (source.get("derived") or {}).get("transition")
    return result


def run_rewritten_context_eval(config: dict[str, Any]) -> list[dict[str, Any]]:
    input_cfg = config.get("input", {})
    output_path = config.get("output", {}).get("result_path", "data/generations/rewritten_context_qwen3_5_9b_results.jsonl")
    rewrite_path = input_cfg["rewrite_records_path"]
    split_filter = set(input_cfg.get("split_filter") or [])
    dataset_filter = set(input_cfg.get("datasets") or [])
    require_success = bool(input_cfg.get("require_rewrite_success", True))
    max_examples = input_cfg.get("max_examples")

    rows = read_jsonl(rewrite_path)
    scope_path = scope_path_from_config(config, "input", "runtime")
    if scope_path:
        scope_ids = load_scope_ids(scope_path)
        before = len(rows)
        rows = [r for r in rows if _result_key(r) in scope_ids]
        log(f"Applied fixed sample scope: {len(rows)} / {before} rewrite records kept from {scope_path}")
    if require_success:
        rows = [r for r in rows if r.get("rewrite_status") == "success" and r.get("rewritten_context")]
    if dataset_filter:
        rows = [r for r in rows if r.get("dataset") in dataset_filter]
    if split_filter:
        rows = [r for r in rows if r.get("split") in split_filter]
    if max_examples:
        rows = rows[:int(max_examples)]

    source_cache = _load_source_cache(rows)
    existing = read_jsonl(output_path) if bool(config.get("runtime", {}).get("resume", True)) else []
    existing_by_id = {str((r.get("example") or {}).get("id") or ""): r for r in existing if r.get("example")}
    pending = [r for r in rows if _result_key(r) not in existing_by_id]

    log(f"Loaded rewrite records: {len(rows)}")
    log(f"Existing completed results: {len(existing_by_id)}")
    log(f"Pending rewritten-context eval: {len(pending)}")

    model_llm = build_llm(config["model"])
    judge_llm = build_llm(config["judge_model"])
    runtime = config.get("runtime", {})
    num_workers = int(runtime.get("num_workers", 16))
    save_every = int(runtime.get("save_every", 100))

    results = list(existing_by_id.values())
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_evaluate_one, row, _source_record(row, source_cache), model_llm, judge_llm, config): row
            for row in pending
        }
        for count, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Rewritten eval"), start=1):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                log(f"Error processing {row.get('example_id')}: {exc}")
            if count % save_every == 0:
                atomic_write_jsonl(results, output_path)
                log(f"Saved progress to {output_path}")

    ordered = {str((r.get("example") or {}).get("id") or ""): r for r in results}
    output_records = [ordered[_result_key(r)] for r in rows if _result_key(r) in ordered]
    atomic_write_jsonl(output_records, output_path)
    log(f"Done. Saved {len(output_records)} results to {output_path}")
    return output_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_rewritten_context_eval(load_yaml(args.config))


if __name__ == "__main__":
    main()
