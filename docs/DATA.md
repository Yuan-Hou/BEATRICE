# Data

All files are xz-compressed JSONL; run `bash scripts/decompress_data.sh` in
`beatrice/` first. Source datasets retain their original licenses (MuSiQue,
HotpotQA, 2WikiMultihopQA, PopQA, Natural Questions, TriviaQA); the files here
are filtered subsets with model generations and judge labels attached.

## beatrice/data/generations/  (Qwen3.5-9B)

`<dataset>_qwen3_5_9b_results.jsonl` — one row per example, the package's
canonical labeled data. Row schema:

```
example          id, dataset, split, question, contexts[], answers[],
                 answerable, metadata
model.name       generating model
runs.without_context / runs.with_context
                 extracted_answer   model answer (direct-answer format)
                 is_correct         judge label after the three-vote audit
                 judge_parse_failed
derived          transition (YY/YN/NY/NN), is_improved, is_regressed, ...
judge_consensus  audit agreement metadata (per-run agreement, confidence,
                 issue_type, label_changed)
```

`rewritten_context_qwen3_5_9b_results.jsonl` — rewritten-context augmentation
rows used by the with-context probes (same schema).

These files are the direct input of the feature-extraction configs
(`input_result_path`). Prompts and raw generations are not stored: prompts are
rebuilt deterministically by the extraction code, and no downstream stage
consumes the raw generation text (only `extracted_answer`).

`data/features_judge_consensus/purification_summary.json` — unanimity
purification statistics for the Qwen labels.

## beatrice/data/gemma4/  (Gemma 4 E4B-it)

- `generations_final/` — generation + three-vote judge results, in the exact
  sharded layout the Gemma feature configs reference. Same row schema as
  above, plus `runs.*.judge_vote` (per-vote detail).
- `trainfmt_answers/` — training-format teacher-forcing answers
  (`answer_source: file` in the feature configs).
- `labels/final/` — final consensus labels per dataset and the summary files.

## Scopes and splits

`beatrice/configs/sampling/fixed_scope_ids.json` (Qwen) and
`gemma4_scope_ids.json` (Gemma) pin every split by example id: 30k train /
1.5k validation / 1.5k test in domain (MuSiQue, HotpotQA, 2WikiMultihopQA),
1k PopQA OOD, plus NQ and TriviaQA OOD evaluation sets. Training and
evaluation code resolve membership through these files only; the raw `split`
field in the rows is not authoritative. The scope files also record which
rewritten-context augmentation rows are included in training.

## PGIR evaluation data

`pgir/datasets/<dataset>/{dev,test}.jsonl` — the same canonical splits in
FlashRAG format (built by `pgir/data_prep.py` from the scope files). System
experiments evaluate on the canonical 1.5k test and OOD sets, never on
FlashRAG's native splits (which would overlap probe training data).
