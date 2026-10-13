# Supplementary Code and Data

Code, data, and trained checkpoints for the paper's probe (BEATRICE) and its
retrieval-control application (PGIR). Prepared for anonymous review.

## Layout

```
beatrice/            probe pipeline: labeling, feature extraction, training,
                     fusion search, evaluation, HTTP deployment
  src/ragprobe/      library code
  scripts/           pipeline entry points (all config-driven)
  configs/           experiment configs for both base models
  data/              filtered, judge-labeled datasets (xz-compressed JSONL)
  checkpoints/       champion probe checkpoints (Qwen3.5-9B and Gemma 4 E4B-it)
pgir/                probe-guided iterative RAG (FlashRAG host)
docs/                PIPELINE.md, DATA.md, CHECKPOINTS.md
```

## Environment

Python 3.11+, CUDA GPU (features/serving fit in 24 GB for the 9B model).

```
cd beatrice
pip install -r requirements.txt
pip install -e .
bash scripts/decompress_data.sh     # restores data/**.jsonl from .xz
```

PGIR additionally needs FlashRAG (see `pgir/README.md`) and faiss.

## External resources (not included, all public)

- Base models: `Qwen/Qwen3.5-9B`, `google/gemma-4-E4B-it` (Hugging Face).
- Judge model for label construction: `Qwen/Qwen3.5-27B` served via vLLM
  (only needed to rebuild labels from scratch; shipped data is already labeled).
- PGIR retrieval stack: `intfloat/e5-base-v2` and the FlashRAG `wiki18_100w`
  corpus with its prebuilt Flat index (FlashRAG public releases).
- Third-party baseline implementations (SelfCheckGPT, Lookback-Lens, ReDeEP,
  LUMINA, EigenScore, and related) are not vendored; the runner scripts under
  `beatrice/scripts/run_*` state which official repository each one expects.

## Reproduction levels

Each level is independent and uses only package contents plus the public
resources above. Full commands are in `docs/PIPELINE.md`.

1. **Score with shipped champions (1 GPU).** Start the probe service on the
   shipped checkpoints and score (question, context, answer) triples:
   `python scripts/serve_probe_scorer.py --config configs/experiment/services/serve_probe_scorer_qwen.yaml`
2. **Evaluate champions on the paper's test splits (1 GPU, hours).** Re-extract
   features for the evaluation splits from the shipped labeled data, then run
   the evaluation configs. Reproduces the main AUROC table.
3. **Retrain probes (1-3 GPUs, days).** Re-extract features for all splits,
   train the 15 per-setting signal-combination encoders, run greedy-k fusion
   search, evaluate. Reproduces the ablation and fusion-search tables.
4. **Rebuild labels from scratch (adds vLLM + judge model).** Regenerate
   answers, run the three-prompt judge audit with unanimity purification.
   Shipped data lets you skip this stage; it is included for completeness.
5. **PGIR system experiments (1 GPU + retrieval stack download).** Run the
   FlashRAG baselines and the probe-guided controller on the paper's
   evaluation splits. See `pgir/README.md`.

## Notes

- All training/evaluation example scopes are pinned by id files under
  `beatrice/configs/sampling/` (30k train / 1.5k validation / 1.5k test in
  domain, 1k PopQA OOD, plus NQ and TriviaQA OOD evaluation sets).
- Random seeds are set in the training configs; numbers reproduce to the
  seed-variation level reported in the paper (about ±0.003 AUROC).
- Data files are xz-compressed to fit the upload limit; every consumer path
  in the configs refers to the decompressed names.
