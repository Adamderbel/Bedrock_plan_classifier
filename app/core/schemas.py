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


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = " ".join(html.unescape(str(value)).replace("\xa0", " ").split())
    # Models sometimes wrap values in quotes: '"E-102"' -> 'E-102'.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return None if value.lower() in PLACEHOLDERS else value


def _is_wordy(item: str) -> bool:
    """True for real entries; false for fragments like '2x4', '4' or digit soup."""
    letters = sum(ch.isalpha() for ch in item)
    chars = sum(not ch.isspace() for ch in item)
    return letters >= 2 and letters / chars >= 0.4


class PlanFeatures(BaseModel):
    """What the extractor model reads from a page image (step 2)."""

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
    preliminary_label: Label = Field(
        description="Best-guess class, from the sheet title, sheet prefix and the system drawn."
    )

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
        """Drop placeholders, duplicates and fragments without words (e.g. '2x4', '4')."""
        items, seen = [], set()
        for raw in value or []:
            item = _clean_text(raw)
            if not item or not _is_wordy(item) or item.lower() in seen:
                continue
            seen.add(item.lower())
            items.append(item)
        return items[:MAX_LIST_ITEMS]

    @model_validator(mode="after")
    def _consistency(self) -> PlanFeatures:
        # Symbols copied from the legend would count twice in the embedding text.
        legend = {item.lower() for item in self.legend_items}
        self.detected_symbols = [s for s in self.detected_symbols if s.lower() not in legend]
        # A non-plan has no sheet number or scale; values there are hallucinated.
        if self.preliminary_label == "not_a_plan" and not self.has_title_block:
            self.sheet_title = None
            self.sheet_number = None
            self.scale = None
        return self

    def to_embedding_text(self) -> str:
        """Text that gets embedded for retrieval.

        Used identically for KB examples and for new pages so their vectors are
        comparable. Left out on purpose:
        - the preliminary label, otherwise the search would match label words
          instead of visual content;
        - the full title-block text: its project / client / firm boilerplate is
          shared by every sheet of a project, so pages would match by building
          instead of by discipline. Only the sheet title is kept.
        """
        parts = [f"Description: {self.description}"]
        if self.sheet_title:
            parts.append(f"Sheet title: {self.sheet_title}")
        if self.sheet_number:
            parts.append(f"Sheet number: {self.sheet_number}")
        if self.scale:
            parts.append(f"Scale: {self.scale}")
        if self.legend_items:
            parts.append("Legend: " + ", ".join(self.legend_items))
        if self.detected_symbols:
            parts.append("Symbols: " + ", ".join(self.detected_symbols))
        if not self.has_title_block:
            parts.append("No title block.")
        return "\n".join(parts)


class Classification(BaseModel):
    """Final decision for one page (step 4)."""

    is_plan: bool = Field(description="False only when label is not_a_plan.")
    label: Label
    confidence: float = Field(ge=0, le=1, description="Probability that the label is correct, 0-1.")
    evidence: list[str] = Field(
        min_length=1,
        description="2-5 short facts visible on the page that support the label, e.g. "
        "'title block: FIRST FLOOR PLUMBING PLAN', 'sheet number P-1', 'legend lists CW/HW lines'.",
    )

    @model_validator(mode="after")
    def _plan_flag_matches_label(self) -> Classification:
        if self.is_plan != (self.label != "not_a_plan"):
            raise ValueError("is_plan must be false exactly when label is 'not_a_plan'")
        return self


class RetrievedRef(BaseModel):
    """One retrieved KB document, as reported in the output."""

    type: Literal["rule", "example"]
    id: str | None = None  # rule: the discipline label
    file: str | None = None  # example: file name
    label: Label | None = None  # example: its label
    source: str | None = None  # example: where it came from
    score: float | None = None


class ModelIds(BaseModel):
    extract: str
    decide: str  # model ID, or "fast_path"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class PageResult(BaseModel):
    """API / CLI output for one page."""

    file: str
    page: int = Field(ge=1)
    is_plan: bool | None = None
    label: Label | None = None
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedRef] = Field(default_factory=list)
    model_ids: ModelIds | None = None
    latency_ms: int = 0

    route: Literal["decider", "fast_path", "extractor"] | None = None  # extractor: evaluation baseline only
    use_rag: bool | None = None
    needs_review: bool = False
    usage: dict[str, Usage] = Field(default_factory=dict)  # keyed by step: "extract", "decide"
    error: str | None = None
