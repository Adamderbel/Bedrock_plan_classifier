"""Run one evaluation scenario on the test images and / or the multi-page bundle.

Usage:
    python -m scripts.eval.run_eval --scenario rag_decider
    python -m scripts.eval.run_eval --scenario norag_decider --set images
    python -m scripts.eval.run_eval --scenario rag_decider norag_decider
    python -m scripts.eval.run_eval --scenario all
    python -m scripts.eval.run_eval --list
then compare every scenario run so far:
    python -m scripts.eval.report

Scenarios:
    rag_decider         extract -> retrieve -> decide
    norag_decider       extract -> decide
    extractor_only      extract; the extractor's preliminary label is the answer
    rag_fastpath        extract -> retrieve -> fast path, decider only when signals disagree
    title_hidden_rag    rag_decider with the title-block corner blanked out
    title_hidden_norag  norag_decider with the title-block corner blanked out

Sets:
    images   data/test/<label>/*   24 single-page images, label = folder
    bundle   data/test_multipage/bundle.pdf, labels in bundle.json
    all      both (default)

Model outputs are cached per page and step in scripts/eval/results/_cache/, so
scenarios share the same extraction (fair comparison) and reruns cost nothing.
Delete the cache to force new model calls.

Outputs in scripts/eval/results/<scenario>/:
    images.json / bundle.json   per page: expected label + full PageResult
    metrics.json                metrics per set
    confusion_<set>.png         confusion matrix
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score  # noqa: E402

from app.core.config import ROOT, get_settings  # noqa: E402
from app.core.schemas import LABELS, Classification, ModelIds, PageResult, PlanFeatures, Usage  # noqa: E402
from app.pipeline.decide import decide  # noqa: E402
from app.pipeline.extract import extract_features  # noqa: E402
from app.pipeline.main import classify_page  # noqa: E402
from app.pipeline.render import PageImage, render_pages  # noqa: E402
from app.pipeline.retrieval import retrieve  # noqa: E402
from app.services.cost import summarize  # noqa: E402

RESULTS = ROOT / "scripts" / "eval" / "results"
CACHE = RESULTS / "_cache"
TEST_DIR = ROOT / "data" / "test"
BUNDLE = ROOT / "data" / "test_multipage" / "bundle.pdf"
BUNDLE_LABELS = ROOT / "data" / "test_multipage" / "bundle.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
WORKERS = 3
HIGH_CONFIDENCE = 0.8


@dataclass(frozen=True)
class Scenario:
    mode: str  # "decider" | "fast_path" | "extractor"
    use_rag: bool
    hide_title: bool = False


SCENARIOS = {
    "rag_decider": Scenario("decider", use_rag=True),
    "norag_decider": Scenario("decider", use_rag=False),
    "extractor_only": Scenario("extractor", use_rag=False),
    "rag_fastpath": Scenario("fast_path", use_rag=True),
    "title_hidden_rag": Scenario("decider", use_rag=True, hide_title=True),
    "title_hidden_norag": Scenario("decider", use_rag=False, hide_title=True),
}


@dataclass(frozen=True)
class Item:
    key: str  # unique id, used as cache key
    file: str
    page: PageImage
    expected: str


# --- test data ---------------------------------------------------------------


def load_items(which: str, hide_title: bool) -> list[Item]:
    s = get_settings()
    opts = {"dpi": s.render_dpi, "max_pages": s.max_pages, "max_side": s.max_image_side, "hide_title_block": hide_title}
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
    return items


# --- cached steps ------------------------------------------------------------


class Steps:
    """extract / retrieve / decide for one page, with model outputs cached on disk.

    Records each step's original latency, so cached reruns still report the
    latency of a real run.
    """

    def __init__(self, cache_dir: Path, key: str):
        self.dir = cache_dir
        self.key = key
        self.ms: dict[str, int] = {}

    def _cached(self, step: str, run, parse):
        path = self.dir / f"{self.key}.{step}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            start = time.perf_counter()
            value, usage = run()
            data = {
                "value": value.model_dump(),
                "usage": usage.model_dump(),
                "ms": round((time.perf_counter() - start) * 1000),
            }
            path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        self.ms[step] = data["ms"]
        return parse(data["value"]), Usage(**data["usage"])

    def extract(self, page: PageImage):
        return self._cached("extract", lambda: extract_features(page), PlanFeatures.model_validate)

    def retrieve(self, features: PlanFeatures):
        start = time.perf_counter()  # deterministic and nearly free: not cached
        out = retrieve(features)
        self.ms["retrieve"] = round((time.perf_counter() - start) * 1000)
        return out

    def decide(self, page: PageImage, features: PlanFeatures, retrieval):
        step = "decide_rag" if retrieval is not None else "decide_norag"
        return self._cached(step, lambda: decide(page, features, retrieval), Classification.model_validate)


# --- running a scenario ------------------------------------------------------


def run_item(item: Item, sc: Scenario, cache_dir: Path) -> dict:
    steps = Steps(cache_dir, item.key)
    if sc.mode == "extractor":
        result = extractor_only(item, steps)
    else:
        result = classify_page(
            item.page,
            item.file,
            sc.use_rag,
            fast_path=sc.mode == "fast_path",
            extract_fn=steps.extract,
            retrieve_fn=steps.retrieve,
            decide_fn=steps.decide,
            log=False,
        )
        result.latency_ms = sum(steps.ms.values())
    extracted = steps.extract(item.page)[0].preliminary_label if "extract" in steps.ms else None
    return {"id": item.key, "expected": item.expected, "extractor_label": extracted, **result.model_dump(exclude_none=True)}


def extractor_only(item: Item, steps: Steps) -> PageResult:
    s = get_settings()
    result = PageResult(
        file=item.file,
        page=item.page.page,
        use_rag=False,
        route="extractor",
        model_ids=ModelIds(extract=s.sonnet_model_id, decide="none"),
    )
    try:
        features, result.usage["extract"] = steps.extract(item.page)
        result.label = features.preliminary_label
        result.is_plan = result.label != "not_a_plan"
        result.evidence = [f"extractor preliminary label (sheet title: {features.sheet_title})"]
    except Exception as err:
        result.error = f"{type(err).__name__}: {err}"
    result.latency_ms = sum(steps.ms.values())
    return result


def run_set(name: str, sc: Scenario, which: str) -> list[dict]:
    cache_dir = CACHE / ("title_hidden" if sc.hide_title else "normal")
    cache_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(which, sc.hide_title)
    print(f"\n== {name} / {which}: {len(items)} pages")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(lambda it: run_item(it, sc, cache_dir), items))
    for r in rows:
        pred = r.get("label") or "ERROR"
        mark = "ok" if pred == r["expected"] else "XX"
        conf = f"{r['confidence']:.2f}" if r.get("confidence") is not None else "  - "
        extra = f"  ({r['error']})" if r.get("error") else ""
        print(f"[{mark}] {r['id']:16} expected {r['expected']:16} got {pred:16} conf {conf}  {r.get('route')}{extra}")
    return rows


# --- metrics -----------------------------------------------------------------


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def compute_metrics(rows: list[dict]) -> dict:
    y_true = [r["expected"] for r in rows]
    y_pred = [r.get("label") or "error" for r in rows]
    n = len(rows)
    correct = [t == p for t, p in zip(y_true, y_pred)]
    plans = [i for i, t in enumerate(y_true) if t != "not_a_plan"]
    non_plans = [i for i, t in enumerate(y_true) if t == "not_a_plan"]
    confs = [r["confidence"] for r in rows if r.get("confidence") is not None]
    high = [i for i, r in enumerate(rows) if (r.get("confidence") or 0) >= HIGH_CONFIDENCE]
    fast = [i for i, r in enumerate(rows) if r.get("route") == "fast_path"]
    ext = [r.get("extractor_label") for r in rows]
    present = sorted(set(y_true) | set(y_pred) - {"error"}, key=lambda x: LABELS.index(x) if x in LABELS else 99)

    def share(idx: list[int]) -> float | None:
        return round(sum(correct[i] for i in idx) / len(idx), 3) if idx else None

    return {
        "pages": n,
        "accuracy": round(sum(correct) / n, 3) if n else None,
        "macro_f1": round(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0), 3),
        "per_class": classification_report(y_true, y_pred, labels=present, output_dict=True, zero_division=0),
        "errors": sum(1 for r in rows if r.get("error")),
        "non_plan_rejection_rate": share(non_plans),
        "plans_wrongly_rejected": sum(1 for i in plans if y_pred[i] == "not_a_plan"),
        "mean_confidence": round(statistics.mean(confs), 3) if confs else None,
        "high_confidence_pages": len(high),
        "high_confidence_accuracy": share(high),
        "needs_review": sum(1 for r in rows if r.get("needs_review")),
        "fast_path_share": round(len(fast) / n, 3) if n else None,
        "fast_path_accuracy": share(fast),
        "extractor_accuracy": round(sum(e == t for e, t in zip(ext, y_true)) / n, 3) if n else None,
        "corrected_extractor": sum(1 for e, t, p in zip(ext, y_true, y_pred) if e != t and p == t),
        "broke_extractor": sum(1 for e, t, p in zip(ext, y_true, y_pred) if e == t and p != t),
        "latency_ms_p50": percentile([r["latency_ms"] for r in rows], 0.5),
        "latency_ms_p95": percentile([r["latency_ms"] for r in rows], 0.95),
        "cost": summarize(PageResult.model_validate({k: v for k, v in r.items() if k in PageResult.model_fields}) for r in rows),
        "page_order_ok": [r["page"] for r in rows] == sorted(r["page"] for r in rows),
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


# --- main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scenario", nargs="+", choices=[*SCENARIOS, "all"], help="one or more scenarios, or 'all'"
    )
    parser.add_argument("--set", dest="which", choices=["images", "bundle", "all"], default="all")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    args = parser.parse_args()
    if args.list or not args.scenario:
        for name, sc in SCENARIOS.items():
            print(f"{name:20} mode={sc.mode:9} rag={sc.use_rag!s:5} title_hidden={sc.hide_title}")
        return 0 if args.list else 2

    # Run in SCENARIOS order (cache-friendly), each saved to its own folder.
    names = list(SCENARIOS) if "all" in args.scenario else [n for n in SCENARIOS if n in args.scenario]
    for name in names:
        run_scenario(name, args.which)
    return 0


def run_scenario(name: str, which_arg: str) -> None:
    sc = SCENARIOS[name]
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics["scenario"] = {"name": name, **sc.__dict__}

    for which in (["images", "bundle"] if which_arg == "all" else [which_arg]):
        rows = run_set(name, sc, which)
        (out_dir / f"{which}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        m = compute_metrics(rows)
        metrics[which] = m
        save_confusion(rows, f"{name} - {which}", out_dir / f"confusion_{which}.png")
        print(
            f"-- {which}: accuracy {m['accuracy']}  macro-F1 {m['macro_f1']}  errors {m['errors']}  "
            f"non-plan rejection {m['non_plan_rejection_rate']}  "
            f"${m['cost']['usd_per_1000_pages']}/1000 pages  p50 {m['latency_ms_p50']} ms"
        )
        if m["fast_path_share"]:
            print(f"   fast path: {m['fast_path_share']:.0%} of pages, accuracy {m['fast_path_accuracy']}")

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
