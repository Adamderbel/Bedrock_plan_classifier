"""Try steps 2-4 on one file: extract, retrieve, then decide with and without RAG.

Usage:
    python -m scripts.try_decide data/test/plumbing/test_plumb_01.png

Makes 3 model calls per page (1 extraction + 2 decisions), a few cents with Nova.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.core.config import get_settings
from app.pipeline.decide import decide
from app.pipeline.extract import extract_features
from app.pipeline.render import render_pages
from app.pipeline.retrieval import retrieve


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    s = get_settings()
    pages = render_pages(path.read_bytes(), dpi=s.render_dpi, max_pages=s.max_pages, max_side=s.max_image_side)
    for page in pages:
        print(f"===== {path.name} page {page.page}")
        features, u_ext = extract_features(page)
        print(f"extractor guess : {features.preliminary_label}")
        print(f"sheet title     : {features.sheet_title}   sheet number: {features.sheet_number}")

        retrieval = retrieve(features)
        print("rules           : " + ", ".join(f"{h.doc.metadata['id']} {h.score:.2f}" for h in retrieval.rules))
        print("examples        : " + ", ".join(f"{h.doc.metadata['label']} {h.score:.2f}" for h in retrieval.examples))

        for name, ctx in (("with RAG", retrieval), ("without RAG", None)):
            result, usage = decide(page, features, ctx)
            print(f"\n[{name}] {result.label}  confidence {result.confidence:.2f}  "
                  f"(tokens {usage.input_tokens}/{usage.output_tokens})")
            for item in result.evidence:
                print(f"   - {item}")
        print(f"\nextraction tokens {u_ext.input_tokens}/{u_ext.output_tokens}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
