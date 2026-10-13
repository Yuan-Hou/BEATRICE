import time
from openai import OpenAI

# Client-side prompt templates for models served without a tokenizer chat
# template (the vLLM server then rejects /v1/chat/completions). Used with
# api: completions.
PROMPT_STYLES = {
    "gemma_turns": {
        "template": "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "stop": ["<end_of_turn>"],
    },
}


class OpenAICompatibleLLM:
    def __init__(
        self,
        model_name,
        api_key="EMPTY",
        base_url=None,
        enable_thinking=False,
        max_retries=5,
        retry_sleep=3.0,
        timeout=120.0,
        max_tokens=8192,
        max_total_tokens=None,
        context_token_buffer=256,
        temperature=0.0,
        api="chat",
        prompt_style=None,
        stop=None,
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_total_tokens = max_total_tokens
        self.context_token_buffer = int(context_token_buffer)
        self.temperature = temperature
        if api not in ("chat", "completions"):
            raise ValueError(f"Unsupported api mode: {api}")
        self.api = api
        style = PROMPT_STYLES.get(prompt_style) if prompt_style else None
        if prompt_style and style is None:
            raise ValueError(f"Unknown prompt_style: {prompt_style}")
        self.prompt_template = style["template"] if style else None
        self.stop = list(stop) if stop else (list(style["stop"]) if style else None)

    def _resolve_max_tokens(self, prompt: str):
        request_max_tokens = self.max_tokens
        if self.max_total_tokens is not None:
            estimated_prompt_tokens = max(len(prompt) // 3, 1)
            available = int(self.max_total_tokens) - estimated_prompt_tokens - self.context_token_buffer
            request_max_tokens = max(1, available) if request_max_tokens is None else min(int(request_max_tokens), max(1, available))
        return int(request_max_tokens) if request_max_tokens is not None else None

    def generate(self, prompt: str) -> str:
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                max_tokens = self._resolve_max_tokens(prompt)
                if self.api == "completions":
                    rendered = self.prompt_template.format(prompt=prompt) if self.prompt_template else prompt
                    request = {
                        "model": self.model_name,
                        "prompt": rendered,
                        "temperature": self.temperature,
                        "timeout": self.timeout,
                    }
                    if self.stop:
                        request["stop"] = self.stop
                    if max_tokens is not None:
                        request["max_tokens"] = max_tokens
                    response = self.client.completions.create(**request)
                    return response.choices[0].text
                request = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "timeout": self.timeout,
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking
                        }
                    },
                }
                if self.stop:
                    request["stop"] = self.stop
                if max_tokens is not None:
                    request["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**request)
                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                time.sleep(self.retry_sleep * attempt)

        raise RuntimeError(f"LLM generation failed after retries: {last_error}")


def build_llm(config: dict) -> OpenAICompatibleLLM:
    backend = config.get("backend", "openai_compatible")
    if backend != "openai_compatible":
        raise ValueError(f"Unsupported LLM backend: {backend}")

    return OpenAICompatibleLLM(
        model_name=config["name"],
        api_key=config.get("api_key", "EMPTY"),
        base_url=config.get("base_url"),
        enable_thinking=config.get("enable_thinking", False),
        max_retries=config.get("max_retries", 3),
        retry_sleep=config.get("retry_sleep", 2.0),
        max_tokens=config.get("max_tokens", 8192),
        max_total_tokens=config.get("max_total_tokens"),
        context_token_buffer=config.get("context_token_buffer", 256),
        temperature=config.get("temperature", 0.0),
        timeout=config.get("timeout", 120.0),
        api=config.get("api", "chat"),
        prompt_style=config.get("prompt_style"),
        stop=config.get("stop"),
    )
