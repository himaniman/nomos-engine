"""Provider-agnostic LLM adapter.

Any OpenAI-compatible endpoint works by config alone — WaveSpeed's unified API
(the default) exposes DeepSeek, Claude, Gemini and others behind one base_url,
so switching the "physicist" model is a one-line change. A MockClient stands in
for offline tests and dry runs.

Config (env, loaded from .env):
    WAVESPEED_API_KEY / NOMOS_LLM_API_KEY   the key (a local server takes any key)
    NOMOS_LLM_BASE_URL                       default https://llm.wavespeed.ai/v1
    NOMOS_LLM_MODEL                          default deepseek/deepseek-v4-flash
    NOMOS_LLM_THINKING                       0/1: ask a llama.cpp/Qwen server to think
                                             (unset = the server's own default)
    NOMOS_LLM_TIMEOUT                        seconds per request (default 600)
    NOMOS_EMBED_BASE_URL / NOMOS_EMBED_MODEL an embeddings endpoint (`embed`)

A LOCAL server (any base_url that is not WaveSpeed — e.g. the llama.cpp box at
http://llm.pgs.dom:8080/v1 serving qwen3.6-35b-a3b) needs no key, takes Qwen's own
sampling (top_k) and a per-request thinking switch, and is slow: ~10 tokens/s out,
one request at a time. `chat` records what each call cost in `client.last`.
"""

from __future__ import annotations

import os
import time

DEFAULT_BASE_URL = "https://llm.wavespeed.ai/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class LLMError(Exception):
    pass


class MockClient:
    """Returns scripted replies in order; records the messages it was sent."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[Message]] = []
        self.model = "mock"

    def chat(self, messages: list[Message], **_) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise LLMError("MockClient ran out of scripted replies")
        return self.replies.pop(0)


class OpenAICompatClient:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float | None = None,
                 max_retries: int = 2):
        from openai import OpenAI

        base_url = base_url or os.environ.get("NOMOS_LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("NOMOS_LLM_MODEL", DEFAULT_MODEL)
        self.base_url = base_url
        self.local = "wavespeed" not in base_url
        api_key = (api_key or os.environ.get("NOMOS_LLM_API_KEY")
                   or os.environ.get("WAVESPEED_API_KEY") or ("local" if self.local else None))
        if not api_key:
            raise LLMError("no API key found — set WAVESPEED_API_KEY (see .env)")
        # A TIMEOUT, because these loops run unattended for hours. Without one a single
        # stalled connection hangs the whole run silently — and the runs that matter are
        # exactly the long ones nobody is watching. Generous, since a thinking model on
        # a long prompt legitimately takes minutes.
        self.timeout = float(os.environ.get("NOMOS_LLM_TIMEOUT", 600.0)
                             if timeout is None else timeout)
        self._client = OpenAI(base_url=base_url, api_key=api_key,
                              timeout=self.timeout, max_retries=max_retries)
        think = os.environ.get("NOMOS_LLM_THINKING", "").strip()
        self.thinking = None if think == "" else think not in ("0", "false", "no", "off")
        self.last: dict = {}

    def chat(self, messages: list[Message], temperature: float = 1.0,
             max_tokens: int | None = None, thinking: bool | None = None,
             top_p: float | None = None) -> str:
        """One completion. `thinking` overrides NOMOS_LLM_THINKING for this call; on a
        Qwen/llama.cpp server it is `chat_template_kwargs.enable_thinking`. The hidden
        reasoning, token counts and wall time land in `self.last`."""
        think = self.thinking if thinking is None else thinking
        extra = {}
        if think is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": bool(think)}
        if self.local:
            extra["top_k"] = 20                     # Qwen's recommended sampling
        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                top_p=top_p if top_p is not None else (0.95 if think else 0.8)
                if self.local else None,
                extra_body=extra or None,
            )
        except Exception as e:  # network / auth / rate-limit — surface cleanly
            raise LLMError(f"{type(e).__name__}: {e}") from e
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        self.last = {
            "seconds": time.time() - t0, "thinking": think,
            "reasoning": getattr(msg, "reasoning_content", None)
            or (getattr(msg, "model_extra", None) or {}).get("reasoning_content") or "",
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "finish": resp.choices[0].finish_reason,
        }
        content = msg.content
        if not content:
            raise LLMError("empty response from model"
                           + (" (thinking ate max_tokens)" if self.last["reasoning"] else ""))
        return content

    def busy(self) -> bool:
        """True when a llama.cpp server is serving SOMEONE ELSE right now (it has one
        slot; our own calls are sequential, so a busy slot between them is not ours).
        Unknown servers are never busy."""
        if not self.local:
            return False
        import json
        import urllib.request
        root = self.base_url.rstrip("/")
        root = root[:-3] if root.endswith("/v1") else root
        try:
            with urllib.request.urlopen(root + "/slots", timeout=10) as r:
                return any(s.get("is_processing") for s in json.load(r))
        except Exception:
            return False


def load_env(path: str = ".env") -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
    except Exception:
        pass


def embed(texts: list[str], base_url: str | None = None,
          model: str | None = None) -> list[list[float]]:
    """Sentence embeddings from an OpenAI-compatible endpoint (default: the local
    bge-m3 beside the chat model). Used to tell a new idea from a repeated one."""
    from openai import OpenAI
    load_env()
    base_url = base_url or os.environ.get("NOMOS_EMBED_BASE_URL", "http://llm.pgs.dom:8081/v1")
    model = model or os.environ.get("NOMOS_EMBED_MODEL", "bge-m3")
    try:
        r = OpenAI(base_url=base_url, api_key="local", timeout=60,
                   max_retries=1).embeddings.create(model=model, input=list(texts))
    except Exception as e:
        raise LLMError(f"{type(e).__name__}: {e}") from e
    return [d.embedding for d in r.data]


def make_client(mock: list[str] | None = None, **kwargs):
    """MockClient if `mock` replies are given, else a live OpenAI-compatible client."""
    if mock is not None:
        return MockClient(mock)
    load_env()
    return OpenAICompatClient(**kwargs)
