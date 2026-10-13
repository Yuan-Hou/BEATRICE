from string import Template


QA_WITH_CONTEXT = Template("""Some context information might be helpful to answer the question:

$context

Please answer the following question based on your knowledge and the context provided. If you don't know the answer, please respond with "I don't know" as your final answer.

You are required to answer in the following format:

Reasoning: <your reasoning process>
Answer: <your final answer, either the answer to the question or "I don't know", should within one line>

Question: $question""")


QA_WITHOUT_CONTEXT = Template("""Please answer the following question based on your knowledge. If you don't know the answer, please respond with "I don't know" as your final answer.

You are required to answer in the following format:

Reasoning: <your reasoning process>
Answer: <your final answer, either the answer to the question or "I don't know", should within one line>

Question: $question""")


DIRECT_ANSWER_SYSTEM = """Please answer the following question concisely with JUST the core entity.
DO NOT use starting articles like 'The', 'A', or 'An', and do not use full sentences.

Example 1:
Question: Who discovered penicillin?
Direct Answer: Alexander Fleming

Example 2:
Question: What is the capital of France?
Direct Answer: Paris

"""

LEGACY_MUSIQUE_FEATURE_PROMPT = Template("""$context

Please answer the following question based on your knowledge and the above context provided.

Question: $question

Directly answer with the final answer without any explanation or reasoning process:""")


QUESTION_ONLY_FEATURE_PROMPT = Template("""Please answer the following question based on your knowledge.

Question: $question

Directly answer with the final answer without any explanation or reasoning process:""")


def format_context(contexts):
    return "\n".join("- " + c["text"] for c in contexts)


def build_qa_prompt(example, with_context: bool):
    question = example["question"]
    if with_context:
        context = format_context(example["contexts"])
        return QA_WITH_CONTEXT.substitute(context=context, question=question)
    return QA_WITHOUT_CONTEXT.substitute(question=question)
