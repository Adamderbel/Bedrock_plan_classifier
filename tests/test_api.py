"""API behaviour with the pipeline replaced by a fake (no AWS calls)."""

import dataclasses

import pytest
from fastapi.testclient import TestClient

from app import app as api
from app.core.config import get_settings
from app.core.schemas import ModelIds, PageResult


def _page(n=1, **kw):
    data = {
        "file": "x",
        "page": n,
        "is_plan": True,
        "label": "electrical",
        "confidence": 0.9,
        "evidence": ["sheet E-1"],
        "model_ids": ModelIds(extract="m1", decide="m2"),
        "route": "decider",
    }
    return PageResult(**(data | kw))


@pytest.fixture
def client(monkeypatch):
    calls = {}

    def fake_classify(data, filename, *, use_rag=None, fast_path=None):
        calls.update(filename=filename, use_rag=use_rag, fast_path=fast_path)
        return calls.get("results", [_page()])

    monkeypatch.setattr(api, "classify_file", fake_classify)
    monkeypatch.setattr(api, "get_index", lambda: None)
    with TestClient(api.app) as c:
        c.calls = calls
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["models"]) == {"extract", "decide", "embed"}


def test_classify_image(client, make_image):
    res = client.post("/classify", files={"file": ("plan.png", make_image(), "image/png")})
    assert res.status_code == 200
    (page,) = res.json()
    assert page["label"] == "electrical" and page["page"] == 1
    assert "error" not in page  # None fields are left out, like the spec


def test_query_overrides_are_passed(client, make_image):
    client.post("/classify?use_rag=false&fast_path=true", files={"file": ("p.png", make_image(), "image/png")})
    assert client.calls["use_rag"] is False and client.calls["fast_path"] is True


def test_default_options_come_from_settings(client, make_image):
    client.post("/classify", files={"file": ("p.png", make_image(), "image/png")})
    assert client.calls["use_rag"] is None and client.calls["fast_path"] is None


def test_unsupported_type_is_415(client):
    res = client.post("/classify", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert res.status_code == 415


def test_empty_file_is_422(client):
    res = client.post("/classify", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert res.status_code == 422


def test_too_large_is_413(client, monkeypatch, make_image):
    small = dataclasses.replace(get_settings(), max_upload_mb=0)
    monkeypatch.setattr(api, "get_settings", lambda: small)
    res = client.post("/classify", files={"file": ("big.png", make_image(), "image/png")})
    assert res.status_code == 413


def test_corrupt_pdf_is_422(monkeypatch):
    # Real classify_file: rendering fails before any model call.
    monkeypatch.setattr(api, "get_index", lambda: None)
    with TestClient(api.app) as c:
        res = c.post("/classify", files={"file": ("bad.pdf", b"%PDF-1.4 garbage", "application/pdf")})
    assert res.status_code == 422


@pytest.mark.parametrize(
    "error, status",
    [
        ("ClientError: An error occurred (ThrottlingException) when calling Converse", 503),
        ("ReadTimeoutError: Read timed out", 504),
        ("IndexMissingError: No FAISS index", 503),
        ("StructuredOutputError: Classification not returned", 502),
    ],
)
def test_all_pages_failed_maps_to_status(client, make_image, error, status):
    client.calls["results"] = [_page(1, label=None, is_plan=None, confidence=None, error=error)]
    res = client.post("/classify", files={"file": ("p.png", make_image(), "image/png")})
    assert res.status_code == status
    if status == 503:
        assert res.headers["Retry-After"] == "30"


def test_partial_failure_still_200(client, make_pdf):
    client.calls["results"] = [_page(1), _page(2, label=None, is_plan=None, confidence=None, error="ThrottlingException")]
    res = client.post("/classify", files={"file": ("set.pdf", make_pdf(2), "application/pdf")})
    assert res.status_code == 200
    assert [p.get("error") for p in res.json()] == [None, "ThrottlingException"]


def test_web_ui_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "PlanSense" in res.text
