"""Token cost of classified pages, and the usage log (logs/usage.jsonl)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable

from app.core.config import get_settings
from app.core.schemas import PageResult, Usage

_log_lock = threading.Lock()


def step_prices() -> dict[str, tuple[float, float]]:
    """USD per 1M (input, output) tokens for each pipeline step."""
    s = get_settings()
    return {
        "extract": (s.price_extract_in, s.price_extract_out),
        "embed": (s.price_embed_in, 0.0),
        "decide": (s.price_decide_in, s.price_decide_out),
    }


def usage_cost(usage: dict[str, Usage]) -> float:
    """USD cost of one page's token usage (keyed by step)."""
    prices = step_prices()
    total = 0.0
    for step, u in usage.items():
        price_in, price_out = prices.get(step, (0.0, 0.0))
        total += (u.input_tokens * price_in + u.output_tokens * price_out) / 1_000_000
    return total


def summarize(results: Iterable[PageResult]) -> dict:
    """Token totals per step, total cost, and cost per page / per 1,000 pages."""
    results = list(results)
    tokens: dict[str, Usage] = {}
    for r in results:
        for step, u in r.usage.items():
            t = tokens.setdefault(step, Usage())
            t.input_tokens += u.input_tokens
            t.output_tokens += u.output_tokens
    total = usage_cost(tokens)
    pages = len(results)
    per_page = total / pages if pages else 0.0
    return {
        "pages": pages,
        "tokens": {step: u.model_dump() for step, u in tokens.items()},
        "total_usd": round(total, 4),
        "usd_per_page": round(per_page, 5),
        "usd_per_1000_pages": round(per_page * 1000, 2),
    }


def log_usage(result: PageResult) -> None:
    """Append one line per classified page to logs/usage.jsonl."""
    s = get_settings()
    s.log_dir.mkdir(parents=True, exist_ok=True)
    line = {
        "ts": round(time.time(), 3),
        "file": result.file,
        "page": result.page,
        "route": result.route,
        "use_rag": result.use_rag,
        "label": result.label,
        "latency_ms": result.latency_ms,
        "usage": {step: u.model_dump() for step, u in result.usage.items()},
        "usd": round(usage_cost(result.usage), 6),
        "error": result.error,
    }
    with _log_lock, (s.log_dir / "usage.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
