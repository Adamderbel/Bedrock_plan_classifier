"""Compare every evaluation scenario run so far.

Usage:
    python -m scripts.eval.report

Reads scripts/eval/results/<scenario>/{metrics,images,bundle}.json and writes:
    scripts/eval/results/report.md        comparison tables and key deltas
    scripts/eval/results/predictions.csv  one row per test page, one column per scenario
"""

from __future__ import annotations

import csv
import json
import sys

from app.core.schemas import LABELS
from scripts.eval.run_eval import RESULTS, SCENARIOS

COMPARISONS = [
    ("rag_decider", "norag_decider", "What RAG adds (title visible)"),
    ("title_hidden_rag", "title_hidden_norag", "What RAG adds (title block hidden)"),
    ("norag_decider", "extractor_only", "What the decider adds over the extractor alone"),
    ("rag_fastpath", "rag_decider", "Fast path vs always calling the decider (both with RAG)"),
]


def load() -> dict[str, dict]:
    runs = {}
    for name in SCENARIOS:
        path = RESULTS / name / "metrics.json"
        if path.exists():
            runs[name] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def fmt(value, pct: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if pct:
        return f"{value:.0%}"
    return str(value)


def scenario_table(runs: dict, which: str) -> list[str]:
    rows = [
        "| Scenario | Pages | Accuracy | Macro-F1 | Non-plan rejection | Plans wrongly rejected | "
        "Mean conf. | Acc. when conf ≥ 0.8 | Needs review | Fast path | $ / 1000 pages | p50 latency (s) | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        m = run.get(which)
        if not m:
            continue
        rows.append(
            f"| {name} | {m['pages']} | {fmt(m['accuracy'], True)} | {fmt(m['macro_f1'])} | "
            f"{fmt(m['non_plan_rejection_rate'], True)} | {m['plans_wrongly_rejected']} | "
            f"{fmt(m['mean_confidence'])} | {fmt(m['high_confidence_accuracy'], True)} | {m['needs_review']} | "
            f"{fmt(m['fast_path_share'], True)} | {m['cost']['usd_per_1000_pages']} | "
            f"{m['latency_ms_p50'] / 1000:.1f} | {m['errors']} |"
        )
    return rows


def per_class_table(runs: dict) -> list[str]:
    names = [n for n, r in runs.items() if r.get("images")]
    rows = ["| Class | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label in LABELS:
        cells = []
        for n in names:
            f1 = runs[n]["images"]["per_class"].get(label, {}).get("f1-score")
            cells.append("-" if f1 is None else f"{f1:.2f}")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return rows


def comparisons(runs: dict) -> list[str]:
    out = []
    for a, b, title in COMPARISONS:
        if a in runs and b in runs and runs[a].get("images") and runs[b].get("images"):
            ma, mb = runs[a]["images"], runs[b]["images"]
            d_acc = (ma["accuracy"] - mb["accuracy"]) * 100
            d_cost = ma["cost"]["usd_per_1000_pages"] - mb["cost"]["usd_per_1000_pages"]
            out.append(
                f"- **{title}** ({a} vs {b}): accuracy {d_acc:+.1f} points, "
                f"macro-F1 {ma['macro_f1'] - mb['macro_f1']:+.3f}, cost {d_cost:+.2f} $/1000 pages"
            )
    return out or ["- (run both scenarios of a pair to see its comparison)"]


def bundle_check(runs: dict) -> list[str]:
    out = []
    for name in runs:
        path = RESULTS / name / "bundle.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        pages = " ".join(f"p{r['page']}:{'ok' if r.get('label') == r['expected'] else 'XX'}" for r in rows)
        order = "in order" if [r["page"] for r in rows] == list(range(1, len(rows) + 1)) else "OUT OF ORDER"
        out.append(f"- {name}: {len(rows)} results, {order} — {pages}")
    return out or ["- (no bundle runs yet)"]


def write_predictions(runs: dict) -> None:
    table: dict[str, dict] = {}
    for name in runs:
        for which in ("images", "bundle"):
            path = RESULTS / name / f"{which}.json"
            if not path.exists():
                continue
            for r in json.loads(path.read_text(encoding="utf-8")):
                row = table.setdefault(r["id"], {"id": r["id"], "set": which, "expected": r["expected"]})
                if r.get("extractor_label") and not row.get("extractor"):
                    row["extractor"] = r["extractor_label"]
                row[name] = r.get("label") or "ERROR"
                if r.get("confidence") is not None:
                    row[f"{name}_conf"] = r["confidence"]
    fields = ["id", "set", "expected", "extractor"]
    for name in runs:
        fields += [name, f"{name}_conf"]
    with (RESULTS / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(table.values())


def main() -> int:
    runs = load()
    if not runs:
        print("No scenario results yet; run: python -m scripts.eval.run_eval --scenario rag_decider")
        return 1
    lines = [
        "# PlanSense evaluation",
        "",
        "## Test images (24 pages, 3 per class)",
        "",
        *scenario_table(runs, "images"),
        "",
        "## Key comparisons (test images)",
        "",
        *comparisons(runs),
        "",
        "## F1 per class (test images)",
        "",
        *per_class_table(runs),
        "",
        "## Multi-page PDF bundle (5 pages)",
        "",
        *scenario_table(runs, "bundle"),
        "",
        *bundle_check(runs),
        "",
        "Confusion matrices: `<scenario>/confusion_images.png`. Per-page results: `predictions.csv`.",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_predictions(runs)
    print("\n".join(lines))
    print(f"\nWrote {RESULTS / 'report.md'} and {RESULTS / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
