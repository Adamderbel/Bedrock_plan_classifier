"""Pipeline wiring with the model call replaced by a fake (no AWS calls)."""

import pytest

from app.core.schemas import PlanFeatures, Usage
from app.pipeline import main


def _features(**kw):
    data = {
        "has_title_block": True,
        "sheet_title": "FIRST FLOOR PLUMBING PLAN",
        "sheet_number": "P-1",
        "description": "Plumbing plan.",
        "label": "plumbing",
        "confidence": 0.9,
    }
    return PlanFeatures(**(data | kw))


@pytest.fixture
def fake_model(monkeypatch):
    calls = []
    answer = {"features": _features()}

    def fake_extract(page, model_id):
        calls.append(model_id)
        return answer["features"], Usage(input_tokens=5000, output_tokens=400, calls=1)

    monkeypatch.setattr(main, "extract_features", fake_extract)
    monkeypatch.setattr(main, "log_usage", lambda r: None)
    return calls, answer


def test_one_call_per_page_with_the_settings_model(fake_model, make_image):
    calls, _ = fake_model
    (r,) = main.classify_file(make_image(), "plan.png")
    assert calls == ["test-model"] and r.model_id == "test-model"
    assert r.label == "plumbing" and r.is_plan and r.confidence == 0.9 and not r.needs_review
    assert r.evidence[:2] == ["sheet title: FIRST FLOOR PLUMBING PLAN", "sheet number: P-1"]
    assert r.usage.input_tokens == 5000


def test_model_can_be_overridden(fake_model, make_image):
    calls, _ = fake_model
    (r,) = main.classify_file(make_image(), "plan.png", model_id="apac.amazon.nova-pro-v1:0")
    assert calls == ["apac.amazon.nova-pro-v1:0"] and r.model_id == "apac.amazon.nova-pro-v1:0"


def test_low_confidence_needs_review(fake_model, make_image):
    _, answer = fake_model
    answer["features"] = _features(confidence=0.5)
    (r,) = main.classify_file(make_image(), "plan.png")
    assert r.needs_review  # 0.5 < REVIEW_THRESHOLD 0.6


def test_one_result_per_pdf_page(fake_model, make_pdf):
    results = main.classify_file(make_pdf(pages=3), "set.pdf")
    assert [r.page for r in results] == [1, 2, 3]


def test_page_error_is_reported_not_raised(monkeypatch, make_image):
    def boom(page, model_id):
        raise RuntimeError("Bedrock down")

    monkeypatch.setattr(main, "extract_features", boom)
    monkeypatch.setattr(main, "log_usage", lambda r: None)
    (r,) = main.classify_file(make_image(), "plan.png")
    assert r.error == "RuntimeError: Bedrock down" and r.label is None and r.needs_review
