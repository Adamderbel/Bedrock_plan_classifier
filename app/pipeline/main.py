"""The pipeline: file -> pages -> one model call per page -> PageResult.

Used by the API and the CLI:
    python -m app.pipeline.main data/test/plumbing/test_plumb_01.jpg
    python -m app.pipeline.main bundle.pdf --model apac.amazon.nova-pro-v1:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.config import get_settings
from app.core.schemas import PageResult
from app.pipeline.extract import extract_features
from app.pipeline.render import PageImage, render_pages
from app.services.cost import log_usage, summarize


def classify_page(page: PageImage, filename: str, model_id: str | None = None, *, log: bool = True) -> PageResult:
    """Classify one rendered page. Errors are reported on the page, not raised."""
    s = get_settings()
    model_id = model_id or s.model_id
    start = time.perf_counter()
    result = PageResult(file=filename, page=page.page, model_id=model_id)
    try:
        features, result.usage = extract_features(page, model_id)
        result.is_plan = features.is_plan
        result.label = features.label
        result.confidence = round(features.confidence, 3)
        result.evidence = features.evidence()
        result.needs_review = features.confidence < s.review_threshold
    except Exception as err:  # one bad page must not fail the whole file
        result.error = f"{type(err).__name__}: {err}"
        result.needs_review = True
    result.latency_ms = round((time.perf_counter() - start) * 1000)
    if log:
        log_usage(result)
    return result


def classify_file(data: bytes, filename: str, *, model_id: str | None = None) -> list[PageResult]:
    """Classify every page of a PDF / PNG / JPEG; one PageResult per page, in page order.

    model_id defaults to MODEL_ID in the settings.
    Raises render.RenderError subclasses for unsupported, corrupt or too-long files.
    """
    s = get_settings()
    pages = render_pages(
        data,
        dpi=s.render_dpi,
        max_pages=s.max_pages,
        max_side=s.max_image_side,
        max_image_bytes=s.max_image_bytes,
    )

    def one(p: PageImage) -> PageResult:
        return classify_page(p, filename, model_id)

    if len(pages) == 1:
        return [one(pages[0])]
    with ThreadPoolExecutor(max_workers=s.page_workers) as pool:
        return list(pool.map(one, pages))


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify the pages of a construction document.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--model", help="Bedrock model ID (default: MODEL_ID); prices in .env still apply")
    args = parser.parse_args()

    results = classify_file(args.file.read_bytes(), args.file.name, model_id=args.model)
    print(json.dumps([r.model_dump(exclude_none=True) for r in results], indent=2))
    print(json.dumps(summarize(results)), file=sys.stderr)
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
