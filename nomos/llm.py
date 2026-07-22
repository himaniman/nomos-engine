"""Provider-agnostic LLM adapter.

Any OpenAI-compatible endpoint works by config alone — WaveSpeed's unified API
(the default) exposes DeepSeek, Claude, Gemini and others behind one base_url,
so switching the "physicist" model is a one-line change. A MockClient stands in
for offline tests and dry runs.

Config (env, loaded from .env):
    WAVESPEED_API_KEY / NOMOS_LLM_API_KEY   the key
    NOMOS_LLM_BASE_URL                       default https://llm.wavespeed.ai/v1
    NOMOS_LLM_MODEL                          default deepseek/deepseek-v4-flash
"""

from __future__ import annotations

import os

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
        api_key = (api_key or os.environ.get("NOMOS_LLM_API_KEY")
                   or os.environ.get("WAVESPEED_API_KEY"))
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

    def chat(self, messages: list[Message], temperature: float = 1.0,
             max_tokens: int | None = None) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:  # network / auth / rate-limit — surface cleanly
            raise LLMError(f"{type(e).__name__}: {e}") from e
        content = resp.choices[0].message.content
        if not content:
            raise LLMError("empty response from model")
        return content


def load_env(path: str = ".env") -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
    except Exception:
        pass


def make_client(mock: list[str] | None = None, **kwargs):
    """MockClient if `mock` replies are given, else a live OpenAI-compatible client."""
    if mock is not None:
        return MockClient(mock)
    load_env()
    return OpenAICompatClient(**kwargs)
