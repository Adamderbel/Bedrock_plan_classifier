import pytest
from pydantic import ValidationError

from app.core.schemas import Classification, PageResult, PlanFeatures, RetrievedRef


def _cls(**overrides):
    data = {"is_plan": True, "label": "electrical", "confidence": 0.9, "evidence": ["Sheet E-102"]}
    return Classification(**(data | overrides))


def test_valid_classification():
    c = _cls()
    assert c.label == "electrical"


def test_not_a_plan_is_valid_when_is_plan_false():
    c = _cls(is_plan=False, label="not_a_plan")
    assert not c.is_plan


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 1.2},
        {"confidence": -0.1},
        {"label": "landscape"},
        {"evidence": []},
        {"is_plan": True, "label": "not_a_plan"},
        {"is_plan": False, "label": "plumbing"},
    ],
)
def test_invalid_classification(overrides):
    with pytest.raises(ValidationError):
        _cls(**overrides)


def test_embedding_text_has_key_features_but_not_label_or_project():
    f = PlanFeatures(
        has_title_block=True,
        title_block_text="ACME TOWER - SMITH & PARTNERS ARCHITECTS - LIGHTING LAYOUT LEVEL 1",
        sheet_title="LIGHTING LAYOUT LEVEL 1",
        sheet_number="E-102",
        scale="1:100",
        legend_items=["downlight", "switch"],
        detected_symbols=["circuit tags"],
        description="Floor plan with lighting fixtures.",
        preliminary_label="electrical",
    )
    text = f.to_embedding_text()
    assert "E-102" in text
    assert "LIGHTING LAYOUT LEVEL 1" in text
    assert "downlight, switch" in text
    assert "electrical" not in text
    assert "ACME" not in text


def test_embedding_text_flags_missing_title_block():
    f = PlanFeatures(has_title_block=False, description="Photo of a building site.", preliminary_label="not_a_plan")
    assert "No title block." in f.to_embedding_text()


def test_features_clean_placeholders_and_junk():
    f = PlanFeatures(
        has_title_block=True,
        title_block_text="&nbsp;",
        sheet_number="&quot;E-102&quot;",
        scale=" N/A ",
        legend_items=["2x4", "4", "2x4', 2, '2', '2x4'", "Downlight", "downlight", "", "Exit sign"],
        detected_symbols=["exit sign"] + [f"symbol {i}" for i in range(30)],
        description="Lighting plan.",
        preliminary_label="electrical",
    )
    assert f.title_block_text == ""
    assert f.sheet_number == "E-102"
    assert f.scale is None
    assert f.legend_items == ["Downlight", "Exit sign"]
    assert "exit sign" not in f.detected_symbols
    assert len(f.detected_symbols) <= 15


def test_not_a_plan_drops_hallucinated_sheet_and_scale():
    f = PlanFeatures(
        has_title_block=False,
        sheet_number="189",
        scale="1/8\" = 1'-0\" (implied)",
        description="Invoice.",
        preliminary_label="not_a_plan",
    )
    assert f.sheet_number is None and f.scale is None


def test_page_result_serializes_like_spec():
    r = PageResult(
        file="lot3.pdf",
        page=1,
        retrieved=[
            RetrievedRef(type="rule", id="electrical"),
            RetrievedRef(type="example", file="ex_elec_04.png"),
        ],
    )
    out = r.model_dump(exclude_none=True)
    assert out["retrieved"] == [{"type": "rule", "id": "electrical"}, {"type": "example", "file": "ex_elec_04.png"}]
