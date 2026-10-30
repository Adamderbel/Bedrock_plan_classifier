"""Token cost of classified pages, and the usage log (logs/usage.jsonl)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable

from app.core.config import get_settings
from app.core.schemas import PageResult, Usage

_log_lock = threading.Lock()


def usage_cost(usage: Usage | None) -> float:
    """USD cost of one page's token usage, at the prices in the settings."""
    if usage is None:
        return 0.0
    s = get_settings()
    return (usage.input_tokens * s.price_in + usage.output_tokens * s.price_out) / 1_000_000


def summarize(results: Iterable[PageResult]) -> dict:
    """Token totals, total cost, and cost per page / per 1,000 pages."""
    results = list(results)
    tokens = Usage()
    for r in results:
        if r.usage:
            tokens.input_tokens += r.usage.input_tokens
            tokens.output_tokens += r.usage.output_tokens
            tokens.calls += r.usage.calls
    total = usage_cost(tokens)
    pages = len(results)
    per_page = total / pages if pages else 0.0
    return {
        "pages": pages,
        "tokens": tokens.model_dump(),
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
        "model_id": result.model_id,
        "label": result.label,
        "latency_ms": result.latency_ms,
        "usage": result.usage.model_dump() if result.usage else None,
        "usd": round(usage_cost(result.usage), 6),
        "error": result.error,
    }
    with _log_lock, (s.log_dir / "usage.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
