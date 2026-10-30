"""Evaluation: one model call per page on the test set, for one or more models.

Usage:
    python -m scripts.eval.run_eval                      # reference model, both scenarios
    python -m scripts.eval.run_eval --model all
    python -m scripts.eval.run_eval --model nova-pro kimi-k2.5 --scenario title_visible
    python -m scripts.eval.run_eval --list
then:
    python -m scripts.eval.report

Models come from models.json, scenarios from scenarios.json (edit those, not this file):
    title_visible   the pages as they are
    title_hidden    the title-block corner blanked out, to test the drawing-only judgement

Sets:
    images   data/test/<label>/*   24 single-page images, label = folder
    bundle   data/test_multipage/bundle.pdf, labels in bundle.json

Model outputs are cached per model / scenario / page in results/_cache/, so reruns only
call the model for pages not done yet (errors are not cached). Delete the cache after
changing the prompt or the schema. Outputs per run in results/<model>/<scenario>/:
images.json, bundle.json, metrics.json, confusion_<set>.png.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score  # noqa: E402

from app.core.config import ROOT, get_settings  # noqa: E402
from app.core.schemas import LABELS, PlanFeatures, Usage  # noqa: E402
from app.pipeline.extract import extract_features  # noqa: E402
from app.pipeline.render import PageImage, render_pages  # noqa: E402
from scripts.eval.config import CACHE, RESULTS, Model, Scenario, load_models, load_scenarios, select  # noqa: E402

TEST_DIR = ROOT / "data" / "test"
BUNDLE = ROOT / "data" / "test_multipage" / "bundle.pdf"
BUNDLE_LABELS = ROOT / "data" / "test_multipage" / "bundle.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SETS = ("images", "bundle")
HIGH_CONFIDENCE = 0.8


@dataclass(frozen=True)
class Item:
    key: str
    file: str
    page: PageImage
    expected: str


# --- test data ---------------------------------------------------------------


@lru_cache(maxsize=4)
def load_items(which: str, hide_title: bool) -> tuple[Item, ...]:
    """Rendered test pages, exactly as the pipeline renders them; cached per scenario switch."""
    s = get_settings()
    opts = {
        "dpi": s.render_dpi,
        "max_pages": s.max_pages,
        "max_side": s.max_image_side,
        "max_image_bytes": s.max_image_bytes,
        "hide_title_block": hide_title,
    }
    items: list[Item] = []
    if which == "images":
        for label in LABELS:
            for path in sorted((TEST_DIR / label).iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    (page,) = render_pages(path.read_bytes(), **opts)
                    items.append(Item(path.stem, path.name, page, label))
    else:
        expected = {e["page"]: e["label"] for e in json.loads(BUNDLE_LABELS.read_text(encoding="utf-8"))}
        for page in render_pages(BUNDLE.read_bytes(), **opts):
            items.append(Item(f"bundle_p{page.page}", BUNDLE.name, page, expected[page.page]))
    return tuple(items)


# --- one page ----------------------------------------------------------------


def run_item(item: Item, sc: Scenario, model: Model, cache_dir: Path) -> dict:
    s = get_settings()
    path = cache_dir / f"{item.key}.json"
    row = {"id": item.key, "file": item.file, "page": item.page.page, "expected": item.expected,
           "model": model.key, "scenario": sc.name}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        start = time.perf_counter()
        try:
            features, usage = extract_features(item.page, model.id)
        except Exception as err:  # reported on the page and not cached, so a rerun retries it
            return {**row, "error": f"{type(err).__name__}: {err}", "latency_ms": 0, "usage": {}, "calls": 0,
                    "needs_review": True}
        data = {"features": features.model_dump(), "usage": usage.model_dump(),
                "ms": round((time.perf_counter() - start) * 1000)}
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    features, usage = PlanFeatures.model_validate(data["features"]), Usage(**data["usage"])
    return {
        **row,
        "label": features.label,
        "is_plan": features.is_plan,
        "confidence": features.confidence,
        "evidence": features.evidence(),
        "needs_review": features.confidence < s.review_threshold,
        "latency_ms": data["ms"],
        "usage": usage.model_dump(),
        "calls": usage.calls,
    }


# --- metrics -----------------------------------------------------------------


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def compute_metrics(rows: list[dict], model: Model) -> dict:
    y_true = [r["expected"] for r in rows]
    y_pred = [r.get("label") or "error" for r in rows]
    n = len(rows)
    correct = [t == p for t, p in zip(y_true, y_pred)]
    plans = [i for i, t in enumerate(y_true) if t != "not_a_plan"]
    non_plans = [i for i, t in enumerate(y_true) if t == "not_a_plan"]
    confs = [r["confidence"] for r in rows if r.get("confidence") is not None]
    high = [i for i, r in enumerate(rows) if (r.get("confidence") or 0) >= HIGH_CONFIDENCE]
    present = sorted(set(y_true) | set(y_pred) - {"error"}, key=lambda x: LABELS.index(x) if x in LABELS else 99)
    answered = [r["latency_ms"] for r in rows if not r.get("error")]
    tokens_in = sum(r["usage"].get("input_tokens", 0) for r in rows)
    tokens_out = sum(r["usage"].get("output_tokens", 0) for r in rows)
    usd = model.cost(tokens_in, tokens_out)

    def share(idx: list[int]) -> float | None:
        return round(sum(correct[i] for i in idx) / len(idx), 3) if idx else None

    return {
        "pages": n,
        "accuracy": round(sum(correct) / n, 3) if n else None,
        "macro_f1": round(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0), 3),
        "per_class": classification_report(y_true, y_pred, labels=present, output_dict=True, zero_division=0),
        "all_correct": all(correct),
        "errors": sum(1 for r in rows if r.get("error")),
        "format_failures": sum(1 for r in rows if r.get("calls", 0) > 1),
        "non_plan_rejection_rate": share(non_plans),
        "plans_wrongly_rejected": sum(1 for i in plans if y_pred[i] == "not_a_plan"),
        "mean_confidence": round(statistics.mean(confs), 3) if confs else None,
        "high_confidence_pages": len(high),
        "high_confidence_accuracy": share(high),
        "needs_review": sum(1 for r in rows if r.get("needs_review")),
        "latency_ms_p50": percentile(answered, 0.5),
        "latency_ms_p95": percentile(answered, 0.95),
        "cost": {
            "tokens_in_per_page": round(tokens_in / n) if n else 0,
            "tokens_out_per_page": round(tokens_out / n) if n else 0,
            "total_usd": round(usd, 4),
            "usd_per_page": round(usd / n, 5) if n else 0,
            "usd_per_1000_pages": round(usd / n * 1000, 2) if n else 0,
        },
    }


def save_confusion(rows: list[dict], title: str, path: Path) -> None:
    y_true = [r["expected"] for r in rows]
    y_pred = [r.get("label") or "error" for r in rows]
    labels = list(LABELS) + (["error"] if "error" in y_pred else [])
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- running -----------------------------------------------------------------


def run(model: Model, sc: Scenario, which_arg: str, workers: int) -> None:
    out_dir = RESULTS / model.key / sc.name
    cache_dir = CACHE / model.key / sc.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics.update({"model": model.key, "model_id": model.id, "scenario": sc.name, "hide_title": sc.hide_title})

    for which in (SETS if which_arg == "all" else (which_arg,)):
        items = load_items(which, sc.hide_title)
        print(f"\n== {model.name} / {sc.name} / {which}: {len(items)} pages")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda it: run_item(it, sc, model, cache_dir), items))
        for r in rows:
            pred = r.get("label") or "ERROR"
            mark = "ok" if pred == r["expected"] else "XX"
            conf = f"{r['confidence']:.2f}" if r.get("confidence") is not None else "  - "
            extra = f"  ({r['error']})" if r.get("error") else ("  [retried]" if r.get("calls", 0) > 1 else "")
            print(f"[{mark}] {r['id']:16} expected {r['expected']:16} got {pred:16} conf {conf}{extra}")
        (out_dir / f"{which}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        m = compute_metrics(rows, model)
        metrics[which] = m
        save_confusion(rows, f"{model.name} - {sc.name} - {which}", out_dir / f"confusion_{which}.png")
        print(
            f"-- {which}: accuracy {m['accuracy']}  macro-F1 {m['macro_f1']}  errors {m['errors']}  "
            f"format failures {m['format_failures']}  ${m['cost']['usd_per_1000_pages']}/1000 pages  "
            f"p50 {m['latency_ms_p50']} ms"
        )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> int:
    models, reference = load_models()
    scenarios = load_scenarios()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=[reference], help="model keys from models.json, or 'all'")
    parser.add_argument("--scenario", nargs="+", default=["all"], help="scenario names from scenarios.json, or 'all'")
    parser.add_argument("--set", dest="which", choices=[*SETS, "all"], default="all")
    parser.add_argument("--workers", type=int, default=3, help="pages in parallel (keep within the Bedrock quota)")
    parser.add_argument("--list", action="store_true", help="list models and scenarios and exit")
    args = parser.parse_args()

    if args.list:
        print("Models (models.json):")
        for k, m in models.items():
            ref = "  <- reference" if k == reference else ""
            print(f"  {k:16} {m.name:20} ${m.price_in}/{m.price_out} per 1M  {m.id}{ref}")
        print("Scenarios (scenarios.json):")
        for name, sc in scenarios.items():
            print(f"  {name:16} hide_title={sc.hide_title}")
        return 0

    for model_key in select(args.model, models):
        for sc_name in select(args.scenario, scenarios):
            run(models[model_key], scenarios[sc_name], args.which, args.workers)
    print(f"\nResults in {RESULTS.relative_to(ROOT)}; compare with: python -m scripts.eval.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
