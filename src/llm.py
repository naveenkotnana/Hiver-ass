"""Isolated optional xAI (SpaceXAI) client.

Never required for the demo path. No API keys in source.
Calls are cached on disk, retried, and time-bounded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.paths import DATA, ROOT

logger = logging.getLogger(__name__)

CACHE_PATH = DATA / "cache" / "llm_cache.json"


def llm_available() -> bool:
    return bool(os.getenv("XAI_API_KEY"))


def _cache_load() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _cache_save(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def _key(model: str, system: str, prompt: str) -> str:
    blob = f"{model}\n{system}\n{prompt}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def complete(
    prompt: str,
    *,
    system: str = "",
    model: str | None = None,
    timeout: int = 30,
    max_retries: int = 2,
    temperature: float = 0.0,
) -> str | None:
    """Return model text, or None if no key / all retries failed."""

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return None

    model = model or os.getenv("XAI_MODEL", "grok-4.5")
    base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    cache_key = _key(model, system, prompt)
    cache = _cache_load()
    if cache_key in cache:
        return cache[cache_key]

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed; skipping LLM call")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            cache[cache_key] = text
            _cache_save(cache)
            return text
        except Exception as exc:  # noqa: BLE001 — surface any API/network failure
            last_err = exc
            logger.warning("LLM attempt %s failed: %s", attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))

    logger.error("LLM call failed after retries: %s", last_err)
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from a model response."""

    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


# Silence unused import warning if ROOT is handy for tests.
_ = ROOT
