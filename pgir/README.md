# PGIR: Probe-Guided Iterative RAG

Retrieval-control experiments built on the probe service. FlashRAG hosts the
retrieval stack and baselines; the probe scores candidate answers online and
the controller decides whether to answer, re-retrieve, or rewrite.

## Components

```
gen_prompt.py        training-format direct-answer prompt (imports the probe
                     service code so the prompt is identical by construction)
scorer_client.py     probe service HTTP client (retries, checkpoint pinning)
vllm_client.py       generation client (chat-template token ids sent to
                     /v1/completions of a vLLM OpenAI-compatible server)
pipeline.py          decide() + the PGIR control loop (pool argmax guard)
steer.py             probe-signal evidence actions (s1 / rewrite / s2)
strategies.py        page-turn and answer-guided rewrite strategies with an
                     anti-lock-in guard (Jaccard > 0.6 forces s1)
calibration.py       per-mode isotonic/Platt calibration; fits on dev only and
                     refuses test data; records the probe checkpoint it was
                     fitted against and raises on mismatch
flashrag_adapter.py  FlashRAG Config/Retriever/BasicPipeline integration
run_baselines.py     unified runner: baselines + PGIR, with generation and
                     retrieval call counting
data_prep.py         canonical splits -> FlashRAG-format datasets/
tuning/              full-budget trace replay + (tau_hi, budget) grid scan
analysis/            regression rate, oracle headroom, lock-in, table builders
tests/               unit tests: python -m pgir.tests.run_all
configs/             FlashRAG host configs (E5 + wiki18, local vLLM backend)
datasets/            canonical evaluation splits in FlashRAG format (shipped)
assets/              small shipped inputs (SKR training data); larger baseline
                     artifacts (adaptive labels/classifier, layer probers) are
                     rebuilt by build_adaptive_labels.py, train_adaptive_classifier.py
                     and probingrag/train_probers.py
```

## Setup

1. Install FlashRAG from the public repository at the commit recorded in
   `flashrag_fork/UPSTREAM_COMMIT.txt`, then apply
   `flashrag_fork/local_changes.patch` (one file:
   `flashrag/generator/openai_generator.py`; passes token-id prompts through
   and disables thinking mode for Qwen3.5). Install with
   `pip install --no-deps -e .`.
2. Download the FlashRAG `wiki18_100w` corpus and prebuilt E5 Flat index into
   `pgir/corpus/` and `pgir/indexes/` (public FlashRAG releases), plus
   `intfloat/e5-base-v2`.
3. Start the generation backend (vLLM, OpenAI-compatible, port per
   `configs/flashrag_qwen.yaml`) and the probe service
   (`beatrice/scripts/serve_probe_scorer.py`).

faiss note: retrieval runs on a single thread with BLAS/OMP thread caps
(`pgir/__init__.py` sets them); do not lift these on the search path.

## Running

From the package root:

```
python -m pgir.run_baselines --help     # baselines and PGIR runs
python -m pgir.tuning.run_tuning        # calibration + threshold/budget scan (dev only)
python -m pgir.tests.run_all            # unit tests
```

Evaluation uses `datasets/` (canonical 1.5k test + OOD sets pinned by the
probe scope files), never FlashRAG's native splits, to avoid overlap with
probe training data. Generation is deterministic (temperature 0); Qwen3.5
thinking mode is disabled everywhere.
