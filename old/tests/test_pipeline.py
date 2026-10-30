"""Pipeline wiring with the model calls replaced by fakes (no AWS calls)."""

import pytest
from langchain_core.documents import Document

from app.core.schemas import Classification, PlanFeatures, Usage
from app.pipeline import main
from app.pipeline.retrieval import Hit, Retrieval


@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    calls = {"decide": 0, "retrieve": 0}
    features = PlanFeatures(
        has_title_block=True,
        sheet_title="FIRST FLOOR PLUMBING PLAN",
        sheet_number="P-1",
        description="Plumbing plan.",
        preliminary_label="plumbing",
    )

    def fake_extract(page):
        return features, Usage(input_tokens=5000, output_tokens=400)

    def fake_retrieve(f):
        calls["retrieve"] += 1
        examples = [Hit(Document(page_content="x", metadata={"label": "plumbing", "file": f"{i}.png"}), 0.8) for i in range(4)]
        rules = [Hit(Document(page_content="r", metadata={"id": "plumbing", "source": "kb"}), 0.6)]
        return Retrieval(rules=rules, examples=examples)

    def fake_decide(page, f, retrieval):
        calls["decide"] += 1
        c = Classification(is_plan=True, label="plumbing", confidence=0.5, evidence=["title: PLUMBING PLAN"])
        return c, Usage(input_tokens=8000, output_tokens=150)

    monkeypatch.setattr(main, "extract_features", fake_extract)
    monkeypatch.setattr(main, "retrieve", fake_retrieve)
    monkeypatch.setattr(main, "decide", fake_decide)
    monkeypatch.setattr(main, "log_usage", lambda r: None)
    return calls


def test_decider_route_with_rag(fake_models, make_image):
    (r,) = main.classify_file(make_image(), "plan.png", use_rag=True, fast_path=False)
    assert r.route == "decider" and r.label == "plumbing" and r.use_rag
    assert {ref.type for ref in r.retrieved} == {"rule", "example"}
    assert set(r.usage) == {"extract", "embed", "decide"}
    assert r.needs_review  # confidence 0.5 < 0.6
    assert fake_models["decide"] == 1


def test_no_rag_skips_retrieval(fake_models, make_image):
    (r,) = main.classify_file(make_image(), "plan.png", use_rag=False, fast_path=False)
    assert fake_models["retrieve"] == 0
    assert r.retrieved == [] and "embed" not in r.usage


def test_fast_path_skips_decider(fake_models, make_image):
    (r,) = main.classify_file(make_image(), "plan.png", use_rag=True, fast_path=True)
    assert r.route == "fast_path" and r.model_ids.decide == "fast_path"
    assert fake_models["decide"] == 0 and "decide" not in r.usage


def test_one_result_per_pdf_page(fake_models, make_pdf):
    results = main.classify_file(make_pdf(pages=3), "set.pdf", use_rag=True, fast_path=False)
    assert [r.page for r in results] == [1, 2, 3]


def test_page_error_is_reported_not_raised(fake_models, monkeypatch, make_image):
    def boom(page):
        raise RuntimeError("Bedrock down")

    monkeypatch.setattr(main, "extract_features", boom)
    (r,) = main.classify_file(make_image(), "plan.png", use_rag=True, fast_path=False)
    assert r.error == "RuntimeError: Bedrock down" and r.label is None and r.needs_review
