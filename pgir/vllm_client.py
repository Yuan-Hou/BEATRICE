"""vLLM generation client for the VLLM backend (spec §3.1/§3.2).

Sends chat-templated TOKEN IDS (built by `gen_prompt.prompt_token_ids`, which
reuses the scorer's own `_apply_chat_template`) to vLLM's OpenAI-compatible
/v1/completions endpoint. Passing ids instead of text removes any chance of a
server-side tokenization mismatch and bypasses the chat/reasoning parser.

Decoding is greedy (temperature=0) with max_tokens = the scorer's
self-generation default (32), so the produced answers match the SCORER
backend's answers token-for-token (verified by tests/test_answer_parity.py).
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import httpx

from ragprobe.features.extractor import _apply_chat_template

from .gen_prompt import MAX_ANSWER_TOKENS, load_processor, prompt_token_ids


class VLLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:2333",
        model: str = "Qwen/Qwen3.5-9B",
        concurrency: int = 32,
        timeout_s: float = 180.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.concurrency = int(concurrency)
        self.timeout_s = float(timeout_s)
        self.processor = load_processor(model)
        # HF fast tokenizers are not thread-safe ("Already borrowed") — the
        # client is used from many threads (one per in-flight item)
        self._tok_lock = threading.Lock()

    def generate(self, question: str, contexts=None) -> str:
        return asyncio.run(self.agenerate_many([(question, contexts)]))[0]

    def generate_many(self, qc_pairs: list[tuple[str, Any]]) -> list[str]:
        return asyncio.run(self.agenerate_many(qc_pairs))

    async def agenerate_many(self, qc_pairs: list[tuple[str, Any]]) -> list[str]:
        return [text for text, _ in await self.agenerate_many_timed(qc_pairs)]

    async def agenerate_many_timed(self, qc_pairs: list[tuple[str, Any]]
                                   ) -> list[tuple[str, float]]:
        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:

            async def one(idx: int, question: str, contexts):
                with self._tok_lock:
                    ids = prompt_token_ids(self.processor, question, contexts)
                async with sem:
                    body = {
                        "model": self.model,
                        "prompt": ids,
                        "max_tokens": MAX_ANSWER_TOKENS,
                        "temperature": 0.0,
                    }
                    last_exc: Exception | None = None
                    for attempt in range(3):
                        try:
                            t0 = time.monotonic()
                            r = await client.post(f"{self.base_url}/v1/completions", json=body)
                            r.raise_for_status()
                            text = r.json()["choices"][0]["text"]
                            return idx, text.strip(), time.monotonic() - t0
                        except (httpx.TimeoutException, httpx.TransportError,
                                httpx.HTTPStatusError) as exc:
                            last_exc = exc
                            await asyncio.sleep(2.0 * (attempt + 1))
                    raise RuntimeError(f"vLLM generation failed: {last_exc}")

            results = await asyncio.gather(*(one(i, q, c) for i, (q, c) in enumerate(qc_pairs)))
        out: list[tuple[str, float]] = [("", 0.0)] * len(qc_pairs)
        for idx, text, elapsed in results:
            out[idx] = (text, elapsed)
        return out

    # ---------------------------------------------------- best-of-N sampling ----
    def generate_samples(self, question: str, contexts=None, n: int = 3,
                         temperature: float = 0.8) -> list[str]:
        return asyncio.run(self.agenerate_samples(question, contexts, n, temperature))

    async def agenerate_samples(self, question: str, contexts=None, n: int = 3,
                                temperature: float = 0.8) -> list[str]:
        """n sampled answers for the SAME training-format prompt (best-of-N
        candidates). One request with `n` choices; the greedy path above is
        untouched, so answer parity (tests/test_answer_parity.py) is unaffected."""
        with self._tok_lock:
            ids = prompt_token_ids(self.processor, question, contexts)
        body = {
            "model": self.model,
            "prompt": ids,
            "max_tokens": MAX_ANSWER_TOKENS,
            "temperature": float(temperature),
            "n": int(n),
        }
        return await self._post_completions(body, n_texts=n)

    # ---------------------------------------------------- true continuation ----
    def generate_continuation(self, user_text: str, prefix_text: str = "",
                              max_tokens: int = 64, temperature: float = 0.0) -> str:
        """Continue generating AFTER an assistant prefix (DRAGIN truncate-and-
        resume): ids = chat_template(user_text) + tokenize(prefix_text)."""
        return asyncio.run(self.agenerate_continuation(
            user_text, prefix_text, max_tokens, temperature))

    async def agenerate_continuation(self, user_text: str, prefix_text: str = "",
                                     max_tokens: int = 64,
                                     temperature: float = 0.0) -> str:
        with self._tok_lock:
            ids = _apply_chat_template(self.processor, user_text, "cpu")["input_ids"][0].tolist()
            if prefix_text:
                ids += self.processor.tokenizer.encode(prefix_text,
                                                       add_special_tokens=False)
        body = {"model": self.model, "prompt": ids,
                "max_tokens": int(max_tokens), "temperature": float(temperature)}
        return (await self._post_completions(body, n_texts=1))[0]

    # ------------------------------------------------- free-form generation ----
    def generate_free(self, text_prompt: str, max_tokens: int = 64,
                      temperature: float = 0.0) -> str:
        return asyncio.run(self.agenerate_free(text_prompt, max_tokens, temperature))

    async def agenerate_free(self, text_prompt: str, max_tokens: int = 64,
                             temperature: float = 0.0) -> str:
        """Single-turn free-form generation (STEER query rewrite etc.), chat-
        templated by the same `_apply_chat_template` (thinking stays OFF).
        NOT the training-format answer path — never feed its output to the probe."""
        with self._tok_lock:
            ids = _apply_chat_template(self.processor, text_prompt, "cpu")["input_ids"][0].tolist()
        body = {
            "model": self.model,
            "prompt": ids,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        return (await self._post_completions(body, n_texts=1))[0]

    async def _post_completions(self, body: dict, n_texts: int) -> list[str]:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    r = await client.post(f"{self.base_url}/v1/completions", json=body)
                    r.raise_for_status()
                    choices = r.json()["choices"]
                    return [str(c.get("text", "")).strip() for c in choices[:n_texts]]
                except (httpx.TimeoutException, httpx.TransportError,
                        httpx.HTTPStatusError) as exc:
                    last_exc = exc
                    await asyncio.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"vLLM generation failed: {last_exc}")
