"""The full pipeline: file -> pages -> extract -> retrieve -> route / decide -> PageResult.

Used by the API and the CLI:
    python -m app.pipeline.main data/test/plumbing/test_plumb_01.jpg
    python -m app.pipeline.main bundle.pdf --no-rag --fast-path
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.config import get_settings
from app.core.schemas import ModelIds, PageResult, Usage
from app.pipeline.decide import decide
from app.pipeline.extract import extract_features
from app.pipeline.render import PageImage, render_pages
from app.pipeline.retrieval import retrieve
from app.pipeline.router import fast_path as try_fast_path
from app.services.cost import log_usage, summarize


def classify_page(
    page: PageImage,
    filename: str,
    use_rag: bool,
    fast_path: bool,
    *,
    extract_fn=None,
    retrieve_fn=None,
    decide_fn=None,
    log: bool = True,
) -> PageResult:
    """Classify one rendered page. Errors are reported on the page, not raised.

    The *_fn arguments replace a step (the evaluation passes cached versions);
    by default the real extract / retrieve / decide steps run.
    """
    extract_fn = extract_fn or extract_features
    retrieve_fn = retrieve_fn or retrieve
    decide_fn = decide_fn or decide
    s = get_settings()
    start = time.perf_counter()
    result = PageResult(
        file=filename,
        page=page.page,
        use_rag=use_rag,
        model_ids=ModelIds(extract=s.sonnet_model_id, decide=s.opus_model_id),
    )
    try:
        features, result.usage["extract"] = extract_fn(page)

        retrieval = None
        if use_rag:
            retrieval = retrieve_fn(features)
            result.retrieved = retrieval.refs()
            # Titan's token count is not exposed by LangChain; ~4 characters per token.
            result.usage["embed"] = Usage(input_tokens=len(features.to_embedding_text()) // 4)

        decision = try_fast_path(features, retrieval, s.fast_path_min_agreement) if fast_path else None
        if decision is not None:
            result.route = "fast_path"
            result.model_ids.decide = "fast_path"
        else:
            result.route = "decider"
            decision, result.usage["decide"] = decide_fn(page, features, retrieval)

        result.is_plan = decision.is_plan
        result.label = decision.label
        result.confidence = round(decision.confidence, 3)
        result.evidence = decision.evidence
        result.needs_review = decision.confidence < s.review_threshold
    except Exception as err:  # one bad page must not fail the whole file
        result.error = f"{type(err).__name__}: {err}"
        result.needs_review = True
    result.latency_ms = round((time.perf_counter() - start) * 1000)
    if log:
        log_usage(result)
    return result


def classify_file(
    data: bytes, filename: str, *, use_rag: bool | None = None, fast_path: bool | None = None
) -> list[PageResult]:
    """Classify every page of a PDF / PNG / JPEG; one PageResult per page, in page order.

    use_rag / fast_path default to the settings (USE_RAG, FAST_PATH_ENABLED).
    Raises render.RenderError subclasses for unsupported, corrupt or too-long files.
    """
    s = get_settings()
    use_rag = s.use_rag if use_rag is None else use_rag
    fast_path = s.fast_path_enabled if fast_path is None else fast_path
    pages = render_pages(data, dpi=s.render_dpi, max_pages=s.max_pages, max_side=s.max_image_side)
    if len(pages) == 1:
        return [classify_page(pages[0], filename, use_rag, fast_path)]
    with ThreadPoolExecutor(max_workers=s.page_workers) as pool:
        return list(pool.map(lambda p: classify_page(p, filename, use_rag, fast_path), pages))


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify the pages of a construction document.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--rag", action=argparse.BooleanOptionalAction, default=None, help="override USE_RAG")
    parser.add_argument(
        "--fast-path", action=argparse.BooleanOptionalAction, default=None, help="override FAST_PATH_ENABLED"
    )
    args = parser.parse_args()

    results = classify_file(args.file.read_bytes(), args.file.name, use_rag=args.rag, fast_path=args.fast_path)
    print(json.dumps([r.model_dump(exclude_none=True) for r in results], indent=2))
    print(json.dumps(summarize(results)), file=sys.stderr)
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
