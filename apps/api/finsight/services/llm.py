"""OpenAI LLM client.

Centralized so every agent uses the same configured model, timeouts, and
structured-output handling. Two helpers:

- `chat(messages, ...)` — plain async chat (returns str)
- `chat_structured(messages, schema)` — JSON-schema-validated output
- `stream_chat(messages, ...)` — async generator of token deltas (for streaming
  the final memo to the UI)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from finsight.logging_setup import get_logger
from finsight.settings import settings

log = get_logger(__name__)

_client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=60.0)


def get_client() -> AsyncOpenAI:
    return _client


async def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> str:
    res = await _client.chat.completions.create(
        model=model or settings.openai_chat_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content or ""


async def chat_structured(
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> BaseModel:
    """Constrained JSON output validated against a Pydantic schema.

    Uses OpenAI's `beta.chat.completions.parse` helper which:
      - Converts the Pydantic schema to a strict-mode-compatible JSON schema
        (adds `additionalProperties: false`, promotes all keys to required, etc.)
      - Parses the response into a Pydantic instance.

    Far more reliable than passing `response_format={"type": "json_schema", ...}`
    by hand — the Pydantic→OpenAI strict-schema translation has many edge cases.
    """
    res = await _client.beta.chat.completions.parse(
        model=model or settings.openai_chat_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        response_format=schema,
    )
    parsed = res.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"chat_structured: model refused to produce {schema.__name__}")
    return parsed


async def stream_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> AsyncIterator[str]:
    """Token-by-token streaming used by the writer agent."""
    stream = await _client.chat.completions.create(
        model=model or settings.openai_chat_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


async def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Batch embeddings for RAG ingestion."""
    res = await _client.embeddings.create(
        model=model or settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in res.data]
