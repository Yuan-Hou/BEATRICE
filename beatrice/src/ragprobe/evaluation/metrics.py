from collections import Counter


def transition_from_correctness(without_context_correct: bool, with_context_correct: bool) -> str:
    return ("Y" if without_context_correct else "N") + ("Y" if with_context_correct else "N")


def add_derived_fields(result: dict) -> dict:
    base_correct = result["runs"].get("without_context", {}).get("is_correct", False)
    rag_correct = result["runs"].get("with_context", {}).get("is_correct", False)
    transition = transition_from_correctness(base_correct, rag_correct)

    result["derived"] = {
        "transition": transition,
        "is_improved": transition == "NY",
        "is_regressed": transition == "YN",
        "is_always_correct": transition == "YY",
        "is_always_incorrect": transition == "NN",
    }
    return result


def summarize_transitions(records):
    counter = Counter(r.get("derived", {}).get("transition", "??") for r in records)
    total = sum(counter.values()) or 1
    return {
        key: {
            "count": counter.get(key, 0),
            "ratio": counter.get(key, 0) / total,
        }
        for key in ["NY", "YN", "YY", "NN", "??"]
    }
