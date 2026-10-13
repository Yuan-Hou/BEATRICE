# Pipeline

All commands run from `beatrice/` unless noted. Every stage is config-driven;
the Qwen3.5-9B configs are listed first, the Gemma 4 E4B-it equivalents after.
Stages 1-3 are only needed to rebuild labels from scratch; the shipped data in
`data/` already contains their output.

## 1. Dataset preparation

Source QA datasets (MuSiQue, HotpotQA, 2WikiMultihopQA, PopQA, NQ, TriviaQA)
are normalized into a common JSONL schema (`example` objects with question,
gold contexts, answers):

```
python scripts/prepare_dataset.py --config configs/dataset/<dataset>.yaml
```

## 2. Generation and judging

For each example, the base model answers once without context and once with
the gold context (direct-answer format, vLLM backend). A judge model scores
each answer; three prompt variants vote:

```
python scripts/sample_generations.py --config configs/experiment/generation/<dataset>_qwen9b_eval.yaml
python scripts/audit_generation_labels.py --config configs/experiment/audit/generation_label_audit_qwen27b_voting.yaml
```

Gemma: `configs/experiment/generation/gemma4/`, then
`scripts/gemma/build_gemma_consensus_labels.py` (three-vote consensus) and
`scripts/gemma/gen_trainfmt_answers_vllm.py` (training-format answers used for
teacher forcing; `answer_source: file` in the Gemma feature configs).

## 3. Cleaning (unanimity purification) and scope pinning

Examples where the three judge votes disagree (or parsing fails) are dropped;
the surviving example ids are frozen into scope files that every later stage
respects:

```
python scripts/purify_feature_meta_by_judge_consensus.py --config configs/experiment/audit/purify_judge_consensus_features.yaml
python scripts/create_fixed_sample_scope.py ...   # writes configs/sampling/fixed_scope_ids.json
```

Shipped scopes: `configs/sampling/fixed_scope_ids.json` (Qwen),
`configs/sampling/gemma4_scope_ids.json` (Gemma). Splits: 30k train / 1.5k
validation / 1.5k test in domain, 1k PopQA OOD, plus NQ/TriviaQA OOD sets.
The context-side probes additionally train on a rewritten-context augmentation
(`rewritten_context_*` files); the scope files mark which rewritten rows are
included.

## 4. Feature extraction

One teacher-forced forward per (question, context, answer) triple in the
training prompt format; stores per-token features for the four signal
families: layer-17 hidden states (layer 26 for Gemma), probability scalars,
residual-stream norm and lens channels, attention entropy/mass maps.

Qwen runs in two steps: the base extraction stores last-layer hidden states
plus all other channels; a second pass re-slices hidden states at layer 17 and
appends the residual lens channels, writing the `*_L17_rd` meta/sidecar dirs
that the training configs reference.

```
python scripts/extract_features.py --config configs/experiment/features/context/<dataset>_qwen9b_features.yaml
python scripts/extract_features.py --config configs/experiment/features/question/<dataset>_qwen9b_question_features.yaml
python scripts/extract_withctx_L17rd.py --meta <base meta> --layer 17 --out-root data/features_judge_consensus/qwen3_5_9b_L17_rd          # with-context
python scripts/extract_layer_sidecars.py --meta <base meta> --layer 17 --setting noctx --out-root data/features_judge_consensus/qwen3_5_9b_question_L17_rd   # closed-book (+ extract_residual_direction.py)
```

Gemma (rolling extract -> pack -> verify, per shard):
`bash scripts/gemma/run_gemma_feature_pipeline.sh <LAYER>` with the configs in
`configs/experiment/features/gemma4/`.

For fast training, sidecar features are packed into flat binaries:

```
python scripts/pack_sidecar_features.py ...   # writes data/packed_features/
```

## 5. Encoder training (early fusion, 15 signal combinations)

Per setting (with-context and closed-book), one encoder per non-empty subset
family of the four signal groups: 4 singles, 6 pairs, 4 triples, 1 quad.
Config generators emit the train configs; `train_probe.py` consumes them:

```
python scripts/make_L17rd_attn_configs.py            # closed-book singles + fusions
python scripts/make_L17rd_attn_pairs_configs.py      # pairs
python scripts/make_L17rd_attn_triples_configs.py    # triples + quad
python scripts/make_ctx_L17rd_attn_configs.py        # with-context equivalents
python scripts/train_probe.py --config configs/experiment/probe/<family>/train_<combo>.yaml
```

`scripts/run_L17rd_attn_experiment.sh` and
`scripts/run_ctx_L17rd_attn_experiment.sh` run full campaigns.
Gemma: `scripts/gemma/make_gemma_encoder_configs.py` +
`scripts/gemma/run_gemma_encoder_training.sh`
(configs under `configs/experiment/probe/gemma4_encoders/`).
Note: `train_probe.py` must run with `num_workers: 0` (shared-memory limits).

## 6. Late fusion (greedy-k search over frozen encoders)

Branch encoders are frozen; pooled vectors and branch logits are precomputed
once, then fusion heads over any encoder subset train in seconds. Greedy
forward selection on validation AUC picks the champion subset (k=2 or k=3);
test/OOD numbers are computed for reporting only.

```
python scripts/precompute_frozen_fusion_vecs.py --config <full-fusion config>
python scripts/search_fusion_combos.py ...           # families + greedy selection
python scripts/train_cached_fusion.py --config <champion fusion config>
```

Gemma: `configs/experiment/probe/gemma4_search/pc_{ctx,q}_all15.yaml` then
`train_fusion_{ctx,q}_greedy_k3.yaml`.

## 7. Evaluation

```
python scripts/evaluate_probe.py --config configs/experiment/probe/L17rd_attn/eval_test_L17rd_attn.yaml        # in-domain test
python scripts/evaluate_probe.py --config configs/experiment/probe/L17rd_attn/eval_popqa_L17rd_attn.yaml       # PopQA OOD
python scripts/evaluate_probe.py --config configs/experiment/probe/L17rd_attn/eval_nq_L17rd_attn.yaml          # NQ OOD
python scripts/evaluate_probe.py --config configs/experiment/probe/L17rd_attn/eval_triviaqa_L17rd_attn.yaml    # TriviaQA OOD
```

With-context equivalents under `configs/experiment/probe/ctx_L17rd_attn*/`;
Gemma under `configs/experiment/probe/gemma4_search/eval_*.yaml`.
Baseline runners are `scripts/run_*` (semantic entropy, SelfCheckGPT,
Lookback-Lens, ReDeEP, LUMINA, and the rest); each states the official
third-party repository it expects.

## 8. HTTP deployment

Single-GPU service: online training-format feature extraction -> frozen fusion
probe -> P(answer correct), for closed-book, with-context, and joint scoring:

```
python scripts/serve_probe_scorer.py --config configs/experiment/services/serve_probe_scorer_qwen.yaml
python scripts/smoke_test_probe_scorer.py         # request-level check
python scripts/validate_probe_scorer.py           # AUROC parity vs offline eval
```

Gemma: `serve_probe_scorer_gemma.yaml`. The shipped champion checkpoints under
`checkpoints/` are what these configs load; see `docs/CHECKPOINTS.md`.

## 9. PGIR application

See `pgir/README.md` for the retrieval-control experiments that consume the
probe service.
