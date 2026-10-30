"""Loads models.json and scenarios.json for the evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CACHE = RESULTS / "_cache"


@dataclass(frozen=True)
class Model:
    key: str
    id: str
    name: str
    price_in: float  # USD per 1M input tokens
    price_out: float  # USD per 1M output tokens

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_in + output_tokens * self.price_out) / 1_000_000


@dataclass(frozen=True)
class Scenario:
    name: str
    hide_title: bool


def load_models(path: Path = HERE / "models.json") -> tuple[dict[str, Model], str]:
    """Return ({key: Model}, reference model key)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    models = {k: Model(key=k, **v) for k, v in data["models"].items()}
    reference = data.get("reference", next(iter(models)))
    if reference not in models:
        raise ValueError(f"models.json: reference '{reference}' is not in models")
    return models, reference


def load_scenarios(path: Path = HERE / "scenarios.json") -> dict[str, Scenario]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: Scenario(name=name, **v) for name, v in data.items()}


def select(names: list[str], available: dict) -> list[str]:
    """Names from the command line, or 'all', validated and in config order."""
    if "all" in names:
        return list(available)
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"Unknown: {', '.join(unknown)}. Available: {', '.join(available)}")
    return [n for n in available if n in names]
