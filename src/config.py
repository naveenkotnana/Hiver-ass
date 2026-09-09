"""YAML config + intent taxonomy loader."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.paths import CONFIGS, ROOT

load_dotenv(ROOT / ".env")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    cfg = _read_yaml(CONFIGS / "config.yaml")
    seed = os.getenv("RANDOM_SEED")
    if seed:
        cfg["seed"] = int(seed)
    return cfg


@lru_cache(maxsize=1)
def load_intents() -> dict[str, Any]:
    payload = _read_yaml(CONFIGS / "intents.yaml")
    intents = payload.get("intents") or {}
    if not intents:
        raise ValueError("configs/intents.yaml has no intents")
    return payload


def intent_names() -> list[str]:
    return list(load_intents()["intents"].keys())


def intent_defaults() -> dict[str, str]:
    block = load_intents()["intents"]
    return {name: spec.get("escalation_default", "ESCALATE") for name, spec in block.items()}


def intent_examples() -> dict[str, list[str]]:
    block = load_intents()["intents"]
    return {name: list(spec.get("examples") or []) for name, spec in block.items()}


def intent_keywords() -> dict[str, list[str]]:
    block = load_intents()["intents"]
    return {name: [k.lower() for k in (spec.get("keywords") or [])] for name, spec in block.items()}
