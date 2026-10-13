"""PGIR: Probe-Guided Iterative RAG.

generate -> verify (probe_scorer P(correct)) -> act (accept / retrieve more / argmax).
Everything here is a CLIENT of two running services:

  - probe_scorer  (default http://localhost:8008)  — trained frozen-fusion probe
  - vLLM          (default http://localhost:2333)  — Qwen/Qwen3.5-9B generation

Design contract (spec §0.2-1): answers scored by the probe must come from the
training-time direct-answer prompt with greedy decoding. `pgir.gen_prompt`
therefore *imports* the service-side prompt construction verbatim — there is no
second prompt implementation in this package.

This package lives OUTSIDE the git repo on purpose (the user runs several agents
on the shared checkout and asked for new work to go in a plain new folder).
Run everything from its parent directory, e.g.:

    cd <package_root> && python -m pgir.tests.run_all
"""

import os

# Must precede numpy/faiss import (python -m pgir.* imports this first).
# faiss Flat search from a non-main thread aborts OpenBLAS ("BLAS : Program is
# Terminated ... too many memory regions") when its per-thread buffer pool is
# exhausted by OMP threads x calling threads — cap both.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")

__version__ = "0.1.0"

# override per process to shard datasets across scorer instances (PGIR_SCORER_URL)
PROBE_SCORER_URL = os.environ.get("PGIR_SCORER_URL", "http://localhost:8008")
# PGIR_VLLM_URL / PGIR_MODEL_NAME: backbone migration knobs (gemma campaign
# 2026-07-17). Defaults untouched — Qwen runs stay byte-reproducible.
VLLM_URL = os.environ.get("PGIR_VLLM_URL", "http://localhost:2333")
MODEL_NAME = os.environ.get("PGIR_MODEL_NAME", "Qwen/Qwen3.5-9B")
