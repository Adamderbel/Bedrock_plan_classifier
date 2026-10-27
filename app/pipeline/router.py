"""Step 5 - fast path: answer without the decider when independent signals agree.

Plain code, no model call. A page takes the fast path only if all of these hold:
    1. its sheet number has a known discipline prefix (E-, P-, M-, FP-...)
    2. the extractor's preliminary label is that same discipline
    3. with RAG: at least `min_agreement` of the retrieved examples have that label
and the label is never not_a_plan (rejections always go to the decider).
Anything else returns None and the page goes to the decider.
"""

from __future__ import annotations

import re

from app.core.schemas import Classification, Label, PlanFeatures
from app.pipeline.retrieval import Retrieval

# Sheet-number prefix -> discipline. Ambiguous prefixes are left out on purpose
# (F- is used for both foundations and fire protection, G- for general sheets).
PREFIX_LABELS: dict[str, Label] = {
    "A": "architectural",
    "I": "architectural",
    "S": "structural",
    "E": "electrical",
    "P": "plumbing",
    "M": "hvac_mechanical",
    "H": "hvac_mechanical",
    "C": "civil_site",
    "FP": "fire_safety",
    "SP": "fire_safety",
    "FA": "fire_safety",
    "LS": "fire_safety",
}

# Letters, an optional separator, then a digit: "E-102", "P301", "FP-2", "M 1".
_PREFIX_RE = re.compile(r"^([A-Z]{1,3})[-.\s]?\d")


def prefix_label(sheet_number: str | None) -> tuple[str, Label] | None:
    """Return (prefix, label) for a sheet number with a known discipline prefix."""
    if not sheet_number:
        return None
    match = _PREFIX_RE.match(sheet_number.strip().upper())
    if not match or match.group(1) not in PREFIX_LABELS:
        return None
    return match.group(1), PREFIX_LABELS[match.group(1)]


def fast_path(
    features: PlanFeatures, retrieval: Retrieval | None, min_agreement: int
) -> Classification | None:
    """A Classification if the fast path applies, else None (send to the decider)."""
    found = prefix_label(features.sheet_number)
    if found is None:
        return None
    prefix, label = found
    if features.preliminary_label != label:
        return None

    evidence = [f"sheet number {features.sheet_number} (prefix {prefix}- = {label})"]
    if features.sheet_title:
        evidence.append(f"sheet title: {features.sheet_title}")
    evidence.append(f"page reading independently classified as {label}")

    if retrieval is None:
        # Two agreeing signals: prefix and extractor reading.
        confidence = 0.8
    else:
        labels = retrieval.example_labels()
        agree = labels.count(label)
        if agree < min_agreement:
            return None
        evidence.append(f"{agree}/{len(labels)} most similar reference pages are {label}")
        # Three agreeing signals; more example agreement, more confidence.
        confidence = min(0.95, 0.75 + 0.05 * agree)

    return Classification(is_plan=True, label=label, confidence=confidence, evidence=evidence)
