import pytest
from pydantic import ValidationError

from app.core.schemas import PageResult, PlanFeatures, Usage


def _features(**overrides):
    data = {"has_title_block": True, "description": "Lighting plan.", "label": "electrical", "confidence": 0.9}
    return PlanFeatures(**(data | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 1.2},
        {"confidence": -0.1},
        {"confidence": None},
        {"label": "landscape"},
    ],
)
def test_invalid_features(overrides):
    with pytest.raises(ValidationError):
        _features(**overrides)


def test_model_must_answer_label_and_confidence():
    required = PlanFeatures.model_json_schema()["required"]
    assert {"label", "confidence", "has_title_block", "description"} <= set(required)


def test_is_plan_follows_the_label():
    assert _features().is_plan and not _features(label="not_a_plan").is_plan


def test_features_clean_placeholders_and_junk():
    f = _features(
        title_block_text="&nbsp;",
        sheet_number="&quot;E-102&quot;",
        scale=" N/A ",
        legend_items=["2x4", "4", "2x4', 2, '2', '2x4'", "Downlight", "downlight", "", "Exit sign"],
        detected_symbols=["exit sign"] + [f"symbol {i}" for i in range(30)],
    )
    assert f.title_block_text == ""
    assert f.sheet_number == "E-102"
    assert f.scale is None
    assert f.legend_items == ["Downlight", "Exit sign"]
    assert "exit sign" not in f.detected_symbols
    assert len(f.detected_symbols) <= 15


def test_not_a_plan_drops_hallucinated_sheet_and_scale():
    f = _features(has_title_block=False, sheet_number="189", scale="1/8\" = 1'-0\" (implied)",
                  description="Invoice.", label="not_a_plan")
    assert f.sheet_number is None and f.scale is None


def test_evidence_from_what_was_read():
    f = _features(
        sheet_title="LIGHTING LAYOUT LEVEL 1",
        sheet_number="E-102",
        legend_items=["downlight", "switch", "exit sign", "panel", "outlet"],
        detected_symbols=["circuit tags"],
    )
    assert f.evidence() == [
        "sheet title: LIGHTING LAYOUT LEVEL 1",
        "sheet number: E-102",
        "legend: downlight, switch, exit sign, panel",  # at most 4 entries quoted
        "drawn: circuit tags",
    ]


def test_evidence_without_title_falls_back_to_the_description():
    f = _features(has_title_block=False, description="Photo of a building site.", label="not_a_plan")
    assert f.evidence() == ["no title block", "Photo of a building site."]


def test_page_result_leaves_out_empty_fields():
    r = PageResult(file="lot3.pdf", page=1, label="electrical", confidence=0.9, usage=Usage(input_tokens=10))
    out = r.model_dump(exclude_none=True)
    assert "error" not in out and out["usage"]["input_tokens"] == 10
