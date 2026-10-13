"""Probing-RAG (Baek et al., NAACL 2025 Findings) — method-level reproduction
on our stack (Qwen3.5-9B + E5/wiki18), for the PGIR comparison table.

Original: small FFN probers on mean-pooled mid-layer hidden states decide
whether more retrieval is needed; iterative retrieve-regenerate until the
summed prober logit margin says "no retrieval needed" (threshold theta).

Documented adaptations (backbone swap forces them; method essence preserved):
- backbone Gemma-2B (18L) -> Qwen3.5-9B (32L); candidate layers = even mid
  layers, selected by validation AUROC (their layer choice was validation-driven);
- prompts: our training-format direct answer instead of their CoT few-shot —
  matches the layer-feature assets the probers are trained on (zero drift);
- retriever E5 dense (ours, shared by every system) instead of BM25.
"""
