"""The model call (faked: no AWS calls) and the evaluation config."""

import pytest

from app.core.schemas import PlanFeatures, Usage
from app.pipeline import extract
from app.pipeline.render import render_pages
from scripts.eval.config import load_models, load_scenarios, select


def _features(**kw):
    data = {"has_title_block": False, "description": "Duct layout.", "label": "hvac_mechanical", "confidence": 0.95}
    return PlanFeatures(**(data | kw))


@pytest.mark.parametrize(
    "kw, expected",
    [
        ({}, 0.75),  # plan without a title: capped
        ({"has_title_block": True}, 0.95),  # title visible: unchanged
        ({"sheet_title": "VENTILATION LAYOUT"}, 0.95),
        ({"label": "not_a_plan"}, 0.95),  # non-plans are not capped
        ({"confidence": 0.6}, 0.6),  # already below the cap
    ],
)
def test_confidence_cap_without_title(monkeypatch, make_image, kw, expected):
    monkeypatch.setattr(extract, "invoke_structured", lambda *a, **k: (_features(**kw), Usage(calls=1)))
    (page,) = render_pages(make_image())
    result, usage = extract.extract_features(page, "some-model")
    assert result.confidence == pytest.approx(expected) and usage.calls == 1


def test_model_gets_both_images_and_the_settings_model(monkeypatch, make_image):
    seen = {}

    def fake(model_id, schema, system, content, instruction, **kw):
        seen.update(model=model_id, images=sum(1 for c in content if c["type"] == "image_url"), system=system)
        return _features(has_title_block=True), Usage(calls=1)

    monkeypatch.setattr(extract, "invoke_structured", fake)
    (page,) = render_pages(make_image())
    extract.extract_features(page)
    assert seen["model"] == "test-model" and seen["images"] == 2  # full page + title-block crop
    assert "at most 0.75" in seen["system"]


def test_eval_config_files_are_consistent():
    models, reference = load_models()
    scenarios = load_scenarios()
    assert reference in models and all(m.price_in >= 0 and m.price_out >= 0 for m in models.values())
    assert {sc.hide_title for sc in scenarios.values()} == {False, True}
    assert select(["all"], scenarios) == list(scenarios)
    with pytest.raises(SystemExit):
        select(["nope"], scenarios)
