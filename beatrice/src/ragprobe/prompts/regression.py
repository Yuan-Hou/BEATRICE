from string import Template


REGRESSION_CLASSIFY_TEMPLATE = Template("""# Role and Objective
You are an expert AI researcher evaluating Retrieval-Augmented Generation (RAG) failure modes. Your task is to analyze cases where an LLM answers correctly WITHOUT context, but fails or hallucinates WHEN PROVIDED with context. You must classify the root cause of this failure into one of five predefined categories.

# Category Definitions
1. Attention Bias (Entity/Salience Trap)
- Definition: The model is distracted by highly salient, dense, or physically proximate entities in the context. It mixes up attributes or prioritizes a keyword because it appears frequently or close to the target entity.
- Key marker: Misattributing relationships or getting confused by dense text.

2. Logic Drift (Premature Stop / Overreach)
- Definition: The model fails to complete the necessary multi-hop reasoning chain. It either stops too early or goes one step too far.
- Key marker: Flawed reasoning steps.

3. Hierarchy Misalignment (Granularity Bias)
- Definition: The model selects an answer at the wrong level of abstraction.
- Key marker: Answering with a specific sub-category when a broad category was requested, or vice versa.

4. Over-correction (Faithfulness Paradox)
- Definition: The model becomes overly rigid and refuses to bridge minor gaps using common sense because the context does not explicitly state the connection.
- Key marker: Excessive literalism leading to refusal or denial.

5. Others
- Definition: The error does not fit neatly into the above categories.

# Output Constraints
You must output your entire response in valid JSON format. Do not include markdown or conversational text outside the JSON object.

# JSON Schema
{
  "analysis_summary": "Briefly analyze the difference between the With-Context and Without-Context answers. Explain how the context misled the model.",
  "error_category": "Must be exactly one of: [Attention Bias, Logic Drift, Hierarchy Misalignment, Over-correction, Others]",
  "confidence_score": <int between 1 and 10>,
  "key_distractor_in_context": "Extract the specific 1-2 sentences or entities from the context that derailed the model's reasoning."
}

# Input Data
Please analyze the following case enclosed in <case> tags:

<case>
[Question]: $question
[Correct Answers]: $correct_answers

[Model Answer without Context]:
$answer_without_context

[Model Answer with Context]:
$answer_with_context

[Context]:
$context
</case>""")
