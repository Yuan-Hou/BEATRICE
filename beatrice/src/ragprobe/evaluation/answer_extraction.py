def extract_final_answer(raw_answer: str) -> str:
    if not raw_answer:
        return ""

    if "Answer:" in raw_answer:
        return raw_answer.split("Answer:")[-1].strip().split("\n")[0].strip()

    return raw_answer.strip().split("\n")[-1].strip()


def parse_judge_result(text: str):
    if not text:
        return None

    normalized = text.strip().lower()
    if normalized.startswith("correct"):
        return True
    if normalized.startswith("incorrect"):
        return False
    return None
