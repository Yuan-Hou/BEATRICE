from __future__ import annotations


def format_context(contexts) -> str:
    if isinstance(contexts, str):
        return contexts
    lines = []
    for ctx in contexts or []:
        if isinstance(ctx, dict):
            text = ctx.get("text") or ctx.get("content") or ""
        else:
            text = str(ctx)
        if text:
            lines.append("- " + text)
    return "\n".join(lines)


def answer_with_rag_prompt(question: str, contexts) -> str:
    return f"""Use the context to answer the question. Return exactly one line.

Context:
{format_context(contexts)}

Question: {question}

Output format:
Answer: <one line>"""


def answer_no_rag_prompt(question: str) -> str:
    return f"""Answer the question from your parametric knowledge. Return exactly one line.

Question: {question}

Output format:
Answer: <one line>"""


def evidence_focus_prompt(question: str, contexts) -> str:
    return f"""Extract only the evidence from the context that is most relevant to the question. Keep it short.

Question: {question}

Context:
{format_context(contexts)}

Output format:
Focused Evidence:
- ..."""


def decompose_then_focus_prompt(question: str, contexts) -> str:
    return f"""Decompose the question into subquestions and attach concise supporting evidence from the context.

Question: {question}

Context:
{format_context(contexts)}

Output format:
Subquestions and Evidence:
1. ...
Evidence: ...
2. ...
Evidence: ..."""


def rewrite_context_prompt(question: str, contexts) -> str:
    return f"""Rewrite the question as a better retrieval query, then reorder and compress the existing context around the query. Do not add facts not present in the context.

Question: {question}

Context:
{format_context(contexts)}

Output format:
Rewritten Query: ...
Compressed Context:
- ..."""


def judge_prompt(question: str, answers, model_answer: str) -> str:
    if isinstance(answers, list):
        gold = "; ".join(str(a) for a in answers)
    else:
        gold = str(answers)
    return f"""Judge whether the model answer is semantically correct for the question.

Question: {question}
Gold answer(s): {gold}
Model answer: {model_answer}

Output only one word:
Correct
Incorrect"""




def decompose_question_json_prompt(question: str, *, max_subquestions: int = 3) -> str:
    return f"""Decompose the question into a small directed acyclic graph of subquestions.
Return strict JSON only. Do not wrap it in markdown.

Rules:
- Produce at most {max_subquestions} subquestions.
- The first subquestion must be fully specified and answerable by itself.
- Later subquestions may depend on earlier subquestions and may contain missing components.
- Dependencies must point only to earlier subquestion ids.
- Do not answer the subquestions.

Question: {question}

JSON schema:
{{
  "subquestions": [
    {{"id": "q1", "question": "...", "depends_on": []}},
    {{"id": "q2", "question": "...", "depends_on": ["q1"]}}
  ]
}}"""


def decompose_question_with_context_json_prompt(question: str, contexts, *, max_subquestions: int = 3) -> str:
    return f"""Decompose the question into a small directed acyclic graph of context-grounded subquestions.
Return strict JSON only. Do not wrap it in markdown.

Rules:
- Produce at most {max_subquestions} subquestions.
- The first subquestion must be fully specified and answerable by itself.
- Later subquestions may depend on earlier subquestions and may contain missing components.
- Dependencies must point only to earlier subquestion ids.
- Keep each subquestion focused on evidence available in the context.
- Do not answer the subquestions.

Question: {question}

Context:
{format_context(contexts)}

JSON schema:
{{
  "subquestions": [
    {{"id": "q1", "question": "...", "depends_on": []}},
    {{"id": "q2", "question": "...", "depends_on": ["q1"]}}
  ]
}}"""


def rewrite_child_question_prompt(parent_question: str, prior_answers, child_question: str) -> str:
    if isinstance(prior_answers, list):
        prior = "\n".join(f"- {x}" for x in prior_answers)
    else:
        prior = str(prior_answers)
    return f"""Rewrite the dependent subquestion so it is fully specified and answerable by itself.
Use the prior subquestion answers only to fill missing references. Do not answer the rewritten question.
Return exactly one line.

Original full question: {parent_question}
Prior subquestion answers:
{prior}
Dependent subquestion: {child_question}

Output format:
Question: <fully specified subquestion>"""


def synthesize_from_subanswers_prompt(question: str, subanswers, contexts=None) -> str:
    lines = []
    for item in subanswers or []:
        if isinstance(item, dict):
            lines.append(f"- {item.get('question', '')} => {item.get('answer', '')}")
        else:
            lines.append(f"- {item}")
    context_block = ""
    if contexts:
        context_block = f"\nContext:\n{format_context(contexts)}\n"
    return f"""Answer the original question using the subquestion answers. Return exactly one line.{context_block}
Original question: {question}

Subquestion answers:
{chr(10).join(lines)}

Output format:
Answer: <one line>"""
