from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ragprobe.evaluation.answer_extraction import extract_final_answer, parse_judge_result
from ragprobe.evaluation.metrics import add_derived_fields
from ragprobe.prompts.qa import build_qa_prompt
from ragprobe.prompts.judge import build_judge_prompt, build_judge_prompts
from ragprobe.utils.io import read_jsonl, atomic_write_jsonl
from ragprobe.utils.scope import filter_records_by_scope, load_scope_ids, scope_path_from_config
from ragprobe.utils.logging import log


def _result_key(example_id: str, model_name: str):
    return f"{example_id}::{model_name}"


def _majority_vote(votes: list[bool | None], tie_break: str = "incorrect") -> bool | None:
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    yes = sum(1 for v in valid if v)
    no = len(valid) - yes
    if yes > no:
        return True
    if no > yes:
        return False
    if tie_break == "correct":
        return True
    if tie_break == "abstain":
        return None
    return False


def run_judge(judge_llm, example: dict, extracted_answer: str, config: dict) -> dict:
    judge_cfg = config.get("judge", {})
    voting_cfg = judge_cfg.get("voting", {})
    if not bool(voting_cfg.get("enabled", False)):
        judge_prompt = build_judge_prompt(example["answers"], extracted_answer)
        judge_raw = judge_llm.generate(judge_prompt)
        parsed = parse_judge_result(judge_raw)
        return {
            "judge_raw": judge_raw,
            "is_correct": bool(parsed) if parsed is not None else False,
            "judge_parse_failed": parsed is None,
        }

    variants = voting_cfg.get("variants") or ["strict_semantic", "alias_friendly", "decomposed_fact"]
    tie_break = str(voting_cfg.get("tie_break", "incorrect"))
    prompt_items = build_judge_prompts(example["answers"], extracted_answer, question=example.get("question", ""), variants=variants)
    votes = []
    details = []
    for item in prompt_items:
        raw = judge_llm.generate(item["prompt"])
        parsed = parse_judge_result(raw)
        votes.append(parsed)
        details.append({"variant": item["name"], "raw": raw, "parsed": parsed})
    final = _majority_vote(votes, tie_break=tie_break)
    valid = [v for v in votes if v is not None]
    yes = sum(1 for v in valid if v)
    no = len(valid) - yes
    parse_failed = len(valid) != len(votes)
    return {
        "judge_raw": details[0]["raw"] if details else "",
        "is_correct": bool(final) if final is not None else False,
        "judge_parse_failed": final is None or parse_failed,
        "judge_vote": {
            "enabled": True,
            "variants": details,
            "num_correct_votes": yes,
            "num_incorrect_votes": no,
            "num_parse_failed": len(votes) - len(valid),
            "agreement": max(yes, no) / max(len(valid), 1) if valid else 0.0,
            "tie_break": tie_break,
            "final": final,
        },
    }


def evaluate_one_example(example: dict, model_llm, judge_llm, model_config: dict, eval_config: dict | None = None) -> dict:
    result = {
        "example": example,
        "model": {
            "name": model_config["name"],
            "backend": model_config.get("backend", "openai_compatible"),
            "base_url": model_config.get("base_url"),
        },
        "runs": {},
    }

    run_filter = (eval_config or {}).get("runs")
    for run_name, with_context in [
        ("without_context", False),
        ("with_context", True),
    ]:
        if run_filter and run_name not in run_filter:
            continue
        prompt = build_qa_prompt(example, with_context=with_context)
        raw_answer = model_llm.generate(prompt)
        extracted_answer = extract_final_answer(raw_answer)

        judge_result = run_judge(judge_llm, example, extracted_answer, eval_config or {})

        result["runs"][run_name] = {
            "prompt": prompt,
            "raw_answer": raw_answer,
            "extracted_answer": extracted_answer,
            **judge_result,
        }

    return add_derived_fields(result)


def run_generation_eval(config: dict, model_llm, judge_llm):
    dataset_path = config["dataset_path"]
    result_path = config["output"]["result_path"]
    runtime = config.get("runtime", {})
    num_workers = runtime.get("num_workers", 16)
    save_every = runtime.get("save_every", 100)
    resume = runtime.get("resume", True)

    examples = read_jsonl(dataset_path)
    scope_path = scope_path_from_config(config, "runtime")
    if scope_path:
        before = len(examples)
        examples = filter_records_by_scope(examples, load_scope_ids(scope_path))
        log(f"Applied fixed sample scope: {len(examples)} / {before} examples kept from {scope_path}")
    existing = read_jsonl(result_path) if resume else []
    if scope_path:
        existing = filter_records_by_scope(existing, load_scope_ids(scope_path))

    existing_by_key = {
        _result_key(r["example"]["id"], r["model"]["name"]): r
        for r in existing
        if "example" in r and "model" in r
    }

    model_name = config["model"]["name"]
    pending = [
        ex for ex in examples
        if _result_key(ex["id"], model_name) not in existing_by_key
    ]

    log(f"Loaded {len(examples)} examples.")
    log(f"Existing completed results: {len(existing_by_key)}.")
    log(f"Pending examples: {len(pending)}.")

    results = list(existing_by_key.values())

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(evaluate_one_example, ex, model_llm, judge_llm, config["model"], config): ex
            for ex in pending
        }

        for count, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Evaluating"), start=1):
            ex = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                log(f"Error processing {ex.get('id')}: {e}")

            if count % save_every == 0:
                atomic_write_jsonl(results, result_path)
                log(f"Saved progress to {result_path}")

    atomic_write_jsonl(results, result_path)
    log(f"Done. Saved {len(results)} results to {result_path}")
    return results
