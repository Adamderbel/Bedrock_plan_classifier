"""Step 4 - decision: the decider model gives the final Classification for one page.

Same inputs in both modes, except the retrieved context:
    with RAG     page images + extracted features + retrieved rules and examples
    without RAG  page images + extracted features
so the evaluation measures exactly what retrieval adds.
"""

from __future__ import annotations

import json

from app.core.config import get_settings
from app.core.schemas import Classification, PlanFeatures, Usage
from app.pipeline.extract import LABEL_GUIDE, page_images
from app.pipeline.llm import invoke_structured
from app.pipeline.render import PageImage
from app.pipeline.retrieval import Retrieval

SYSTEM_PROMPT = f"""You are a senior engineer who classifies construction document pages.
For one page you receive the page image, a zoomed crop of its title-block corner, the
features another model extracted from it (they may contain reading errors), and sometimes
reference material retrieved from a labelled knowledge base.

Classes:
{LABEL_GUIDE}

Decide in two steps:
1. Is this page a technical construction drawing at all? Signs: title block, sheet number,
   scale, legend, drawing conventions (linework, dimensions, grid lines, symbols). Photos,
   invoices, letters, forms, maps or app screenshots are not_a_plan. An evacuation plan is a
   plan (fire_safety) even without a title block.
2. If it is a plan, which discipline? In order of strength: the sheet title, the sheet
   number prefix, then the system that is the subject of the drawing. Every discipline is
   drawn over the building outline, so walls and rooms alone do not make it architectural.
   When the prefix and the content disagree, the content wins.

Rules for the answer:
- Use only what is visible on the page. Check the extracted features against the images and
  ignore anything you cannot see.
- Reference rules explain the classes. Reference pages are similar pages from the knowledge
  base with their labels: treat them as hints, not votes, and overrule them when the page's
  own title or content says otherwise.
- evidence: 2-5 short, specific facts read from the page (title, sheet number, legend
  entries, drawn elements) that support THIS label, one fact per item. Room names, walls and
  doors are not evidence for any discipline. Do not paste whole lists. If an important signal
  contradicts the label, add one item saying so, e.g. "sheet prefix M- but the drawing shows
  only sprinkler piping".
- confidence: about 0.9 or more when the title and the drawn content agree; 0.6-0.8 when the
  label rests on the content alone or the scan is hard to read; below 0.6 when the signals
  conflict or the page is mostly unreadable.
- is_plan is false only when the label is not_a_plan."""


def _features_block(features: PlanFeatures) -> dict:
    # The extractor's preliminary label is left out so the decision is an
    # independent second opinion (the fast-path router compares the two).
    data = features.model_dump(exclude={"preliminary_label"})
    return {"type": "text", "text": "Extracted features (may contain reading errors):\n" + json.dumps(data, indent=1)}


def format_context(retrieval: Retrieval) -> str:
    """Retrieved rules and examples as text for the decider."""
    lines = ["REFERENCE RULES (retrieved, most relevant first):"]
    for i, hit in enumerate(retrieval.rules, 1):
        lines += ["", f"--- Rule {i}: {hit.doc.metadata['id']} (similarity {hit.score:.2f}) ---", hit.doc.page_content]
    lines += ["", "SIMILAR REFERENCE PAGES (labelled knowledge-base pages; hints, not votes):"]
    for i, hit in enumerate(retrieval.examples, 1):
        meta = hit.doc.metadata
        lines += [
            "",
            f"--- Page {i}: label={meta['label']} (similarity {hit.score:.2f}, file {meta['file']}) ---",
            hit.doc.page_content,
        ]
    return "\n".join(lines)


def decide(
    page: PageImage, features: PlanFeatures, retrieval: Retrieval | None = None
) -> tuple[Classification, Usage]:
    """Final classification of one page; retrieval=None means no RAG context."""
    content = page_images(page) + [_features_block(features)]
    if retrieval is not None:
        content.append({"type": "text", "text": format_context(retrieval)})
    return invoke_structured(
        get_settings().opus_model_id,
        Classification,
        SYSTEM_PROMPT,
        content,
        "Classify this page.",
        max_tokens=1024,
    )
