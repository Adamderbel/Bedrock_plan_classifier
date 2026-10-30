"""Pydantic data models shared by the pipeline, the API and the evaluation."""

from __future__ import annotations

import html
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

Label = Literal[
    "architectural",
    "structural",
    "electrical",
    "plumbing",
    "hvac_mechanical",
    "civil_site",
    "fire_safety",
    "not_a_plan",
]
LABELS: tuple[str, ...] = get_args(Label)

# Values models emit instead of null; treated as "absent".
PLACEHOLDERS = {"", "&", "&nbsp;", "-", "--", "n/a", "na", "none", "null", "unknown", "unreadable", "not visible"}
MAX_LIST_ITEMS = 15
MAX_EVIDENCE_ITEMS = 4  # legend entries / drawn elements quoted per evidence line


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = " ".join(html.unescape(str(value)).replace("\xa0", " ").split())
    # Models sometimes wrap values in quotes: '"E-102"' -> 'E-102'.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return None if value.lower() in PLACEHOLDERS else value


def _clean_list(value) -> list[str]:
    """Drop placeholders, duplicates and fragments without words (e.g. '2x4', '4')."""
    items, seen = [], set()
    for raw in value or []:
        item = _clean_text(raw)
        if not item or not _is_wordy(item) or item.lower() in seen:
            continue
        seen.add(item.lower())
        items.append(item)
    return items[:MAX_LIST_ITEMS]


def _is_wordy(item: str) -> bool:
    """True for real entries; false for fragments like '2x4', '4' or digit soup."""
    letters = sum(ch.isalpha() for ch in item)
    chars = sum(not ch.isspace() for ch in item)
    return letters >= 2 and letters / chars >= 0.4


class PlanFeatures(BaseModel):
    """What the model reads from a page image, and its classification of the page."""

    has_title_block: bool = Field(description="True if the page has a drawing title block / cartouche.")
    title_block_text: str = Field(
        default="", description="Verbatim text read from the title block (project, sheet title, firm)."
    )
    sheet_title: str | None = Field(
        default=None,
        description="The drawing title from the title block, including any discipline word, e.g. "
        "'ARCHITECTURAL 2nd FLOOR PLAN'; never a room label or project / firm name; null if absent.",
    )
    sheet_number: str | None = Field(
        default=None,
        description="Sheet number as printed, e.g. 'E-102'; prefer the discipline sheet number over "
        "an internal drawing number; null if absent.",
    )
    scale: str | None = Field(default=None, description="Drawing scale as printed, e.g. '1/8\" = 1'-0\"'; null if absent.")
    legend_items: list[str] = Field(
        default_factory=list,
        description="Entries of the legend / symbol key only (not notes or revisions); at most 15.",
    )
    detected_symbols: list[str] = Field(
        default_factory=list,
        description="Distinct drawing elements or symbols seen in the drawing itself, e.g. 'sprinkler heads', "
        "'duct runs', 'rebar callouts'; at most 15.",
    )
    description: str = Field(
        description="2-4 factual sentences naming the drawing type and the systems drawn; no class names."
    )
    label: Label = Field(description="The class, from the sheet title, sheet prefix and the system drawn.")
    confidence: float = Field(ge=0, le=1, description="Probability that the label is correct, 0-1.")

    @field_validator("sheet_title", "sheet_number", "scale", mode="before")
    @classmethod
    def _optional_text(cls, value):
        return _clean_text(value)

    @field_validator("title_block_text", mode="before")
    @classmethod
    def _title_text(cls, value):
        return _clean_text(value) or ""

    @field_validator("legend_items", "detected_symbols", mode="before")
    @classmethod
    def _text_list(cls, value):
        return _clean_list(value)

    @model_validator(mode="after")
    def _consistency(self) -> PlanFeatures:
        # Symbols copied from the legend would be listed twice in the evidence.
        legend = {item.lower() for item in self.legend_items}
        self.detected_symbols = [s for s in self.detected_symbols if s.lower() not in legend]
        # A non-plan has no sheet number or scale; values there are hallucinated.
        if self.label == "not_a_plan" and not self.has_title_block:
            self.sheet_title = None
            self.sheet_number = None
            self.scale = None
        return self

    @property
    def is_plan(self) -> bool:
        return self.label != "not_a_plan"

    @property
    def title_visible(self) -> bool:
        return self.has_title_block or bool(self.sheet_title)

    def evidence(self) -> list[str]:
        """Short facts the label rests on, taken from what the model read on the page."""
        items = []
        if self.sheet_title:
            items.append(f"sheet title: {self.sheet_title}")
        if self.sheet_number:
            items.append(f"sheet number: {self.sheet_number}")
        if self.legend_items:
            items.append("legend: " + ", ".join(self.legend_items[:MAX_EVIDENCE_ITEMS]))
        if self.detected_symbols:
            items.append("drawn: " + ", ".join(self.detected_symbols[:MAX_EVIDENCE_ITEMS]))
        if not self.has_title_block:
            items.append("no title block")
        if len(items) < 2:
            items.append(self.description)
        return items


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0  # model calls made; > 1 means retries or the JSON fallback were needed


class PageResult(BaseModel):
    """API / CLI output for one page."""

    file: str
    page: int = Field(ge=1)
    is_plan: bool | None = None
    label: Label | None = None
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)
    model_id: str | None = None
    latency_ms: int = 0
    needs_review: bool = False
    usage: Usage | None = None
    error: str | None = None
