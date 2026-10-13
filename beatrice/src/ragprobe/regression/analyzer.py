import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ragprobe.prompts.qa import format_context
from ragprobe.prompts.regression import REGRESSION_CLASSIFY_TEMPLATE
from ragprobe.utils.io import read_jsonl, atomic_write_jsonl
from ragprobe.utils.logging import log

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

VALID_CATEGORIES = {
    "Attention Bias",
    "Logic Drift",
    "Hierarchy Misalignment",
    "Over-correction",
    "Others",
}


def normalize_error_category(analysis):
    if not analysis:
        return None

    raw = str(analysis.get("error_category", "Others")).strip().lower()
    if raw.startswith("at"):
        analysis["error_category"] = "Attention Bias"
    elif raw.startswith("lo"):
        analysis["error_category"] = "Logic Drift"
    elif raw.startswith("hi"):
        analysis["error_category"] = "Hierarchy Misalignment"
    elif raw.startswith("ov"):
        analysis["error_category"] = "Over-correction"
    else:
        analysis["error_category"] = "Others"

    return analysis


def parse_json_response(text: str):
    try:
        match = JSON_PATTERN.search(text)
        if match:
            return json.loads(match.group(0))
    except Exception:
        return None
    return None


def analyze_one_regression(result: dict, judge_llm):
    example = result["example"]
    context = format_context(example["contexts"])

    prompt = REGRESSION_CLASSIFY_TEMPLATE.substitute(
        question=example["question"],
        correct_answers=", ".join(example["answers"]),
        answer_without_context=result["runs"]["without_context"].get("raw_answer", ""),
        answer_with_context=result["runs"]["with_context"].get("raw_answer", ""),
        context=context,
    )

    raw = judge_llm.generate(prompt)
    parsed = parse_json_response(raw)
    parsed = normalize_error_category(parsed)

    out = dict(result)
    out["regression_analysis"] = parsed
    out["regression_analysis_raw"] = raw
    return out


def run_regression_analysis(config: dict, judge_llm):
    result_path = config["output"]["result_path"]
    regression_path = config["output"]["regression_path"]
    runtime = config.get("runtime", {})
    num_workers = runtime.get("num_workers", 16)
    save_every = runtime.get("save_every", 50)
    resume = runtime.get("resume", True)

    results = read_jsonl(result_path)
    existing = read_jsonl(regression_path) if resume else []
    existing_ids = {r["example"]["id"] for r in existing if "example" in r}

    regressed = [
        r for r in results
        if r.get("derived", {}).get("transition") == "YN"
        and r["example"]["id"] not in existing_ids
    ]

    records = existing[:]
    log(f"Total results: {len(results)}. Existing regression analyses: {len(existing)}. Pending regressions: {len(regressed)}.")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_one_regression, r, judge_llm): r for r in regressed}

        for count, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Regressions"), start=1):
            try:
                records.append(future.result())
            except Exception as e:
                log(f"Regression analysis error: {e}")

            if count % save_every == 0:
                atomic_write_jsonl(records, regression_path)
                log(f"Saved progress to {regression_path}")

    atomic_write_jsonl(records, regression_path)
    log(f"Done. Saved {len(records)} regression records to {regression_path}")
    return records
