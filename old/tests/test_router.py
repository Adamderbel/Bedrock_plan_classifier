import pytest
from langchain_core.documents import Document

from app.core.schemas import PlanFeatures
from app.pipeline.retrieval import Hit, Retrieval
from app.pipeline.router import fast_path, prefix_label


def _features(sheet_number="E-102", label="electrical", title="LIGHTING PLAN"):
    return PlanFeatures(
        has_title_block=True,
        sheet_title=title,
        sheet_number=sheet_number,
        description="Lighting plan.",
        preliminary_label=label,
    )


def _retrieval(*labels):
    examples = [Hit(Document(page_content="x", metadata={"label": lab, "file": f"{i}.png"}), 0.8) for i, lab in enumerate(labels)]
    return Retrieval(rules=[], examples=examples)


@pytest.mark.parametrize(
    "sheet, expected",
    [
        ("E-102", ("E", "electrical")),
        ("P301", ("P", "plumbing")),
        ("fp-2", ("FP", "fire_safety")),
        ("SP-1", ("SP", "fire_safety")),
        ("M 1", ("M", "hvac_mechanical")),
        ("C-6", ("C", "civil_site")),
        ("03-5908", None),  # internal drawing number
        ("NY-0331", None),  # unknown prefix
        ("F-1", None),  # ambiguous: foundation or fire protection
        ("S-A-2", None),  # no digit after the prefix
        (None, None),
    ],
)
def test_prefix_label(sheet, expected):
    assert prefix_label(sheet) == expected


def test_fast_path_without_rag_needs_prefix_and_extractor_to_agree():
    c = fast_path(_features(), None, min_agreement=3)
    assert c is not None and c.label == "electrical" and c.is_plan
    assert c.confidence == 0.8
    assert fast_path(_features(label="hvac_mechanical"), None, min_agreement=3) is None


def test_fast_path_with_rag_needs_enough_examples():
    agree3 = _retrieval("electrical", "electrical", "plumbing", "electrical")
    c = fast_path(_features(), agree3, min_agreement=3)
    assert c is not None and c.confidence == pytest.approx(0.9)
    assert any("3/4" in e for e in c.evidence)

    agree2 = _retrieval("electrical", "hvac_mechanical", "plumbing", "electrical")
    assert fast_path(_features(), agree2, min_agreement=3) is None


def test_fast_path_skips_conflicting_prefix():
    # Sprinkler sheet numbered M-5: prefix says hvac, reading says fire_safety -> decider.
    f = _features(sheet_number="M-5", label="fire_safety", title="SPRINKLER - SECOND FLOOR PLAN")
    assert fast_path(f, _retrieval(*["fire_safety"] * 4), min_agreement=3) is None


def test_fast_path_never_handles_not_a_plan():
    f = PlanFeatures(has_title_block=False, description="Invoice.", preliminary_label="not_a_plan")
    assert fast_path(f, _retrieval(*["not_a_plan"] * 4), min_agreement=3) is None
