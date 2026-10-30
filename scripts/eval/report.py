"""Compare the evaluation runs of every model.

Usage:
    python -m scripts.eval.report

Reads results/<model>/<scenario>/metrics.json and writes results/report.md and
results/predictions.csv. Ranking: accuracy with the title visible, then accuracy with
the title hidden, then lower cost.
"""

from __future__ import annotations

import csv
import json
import sys

from scripts.eval.config import RESULTS, Model, load_models, load_scenarios

MISSING = "–"
VISIBLE, HIDDEN = "title_visible", "title_hidden"


def load_runs(models: dict[str, Model], scenarios: list[str]) -> dict[tuple[str, str], dict]:
    runs = {}
    for mk in models:
        for sn in scenarios:
            path = RESULTS / mk / sn / "metrics.json"
            if path.exists():
                runs[(mk, sn)] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def metrics(runs: dict, mk: str, sn: str, which: str = "images") -> dict | None:
    return runs.get((mk, sn), {}).get(which)


def pct(x) -> str:
    return MISSING if x is None else f"{x:.0%}"


def num(x, fmt: str = "{}") -> str:
    return MISSING if x is None else fmt.format(x)


def bundle_cell(runs: dict, mk: str, sn: str) -> str:
    b = metrics(runs, mk, sn, "bundle")
    if not b:
        return MISSING
    return f"{round(b['accuracy'] * b['pages'])}/{b['pages']}" + (" ✓" if b["all_correct"] else "")


def rank_key(runs: dict, mk: str):
    m, h = metrics(runs, mk, VISIBLE), metrics(runs, mk, HIDDEN)
    return (m["accuracy"], h["accuracy"] if h else -1.0, -m["cost"]["usd_per_1000_pages"])


def model_table(runs: dict, models: dict[str, Model]) -> list[str]:
    head = ("| Model | Accuracy | Title hidden | Macro-F1 | Non-plans rejected | Plans wrongly rejected | "
            "Mean conf. (visible / hidden) | Acc. when conf ≥ 0.8 | Needs review | Retries | Errors | "
            "Bundle (visible / hidden) | Tokens in / out per page | $ / 1000 pages | p50 / p95 latency (s) |")
    cols = head.count("|") - 1
    lines = [head, "|" + "---|" * cols]
    ranked = sorted((mk for mk in models if metrics(runs, mk, VISIBLE)), key=lambda mk: rank_key(runs, mk), reverse=True)
    for mk in ranked + [mk for mk in models if mk not in ranked]:
        m, h = metrics(runs, mk, VISIBLE), metrics(runs, mk, HIDDEN)
        if not m:
            lines.append(f"| {models[mk].name} | " + " | ".join([MISSING] * (cols - 1)) + " |")
            continue
        c = m["cost"]
        lines.append(
            f"| {models[mk].name} | {pct(m['accuracy'])} | {pct(h['accuracy']) if h else MISSING} | {m['macro_f1']:.2f} | "
            f"{pct(m['non_plan_rejection_rate'])} | {m['plans_wrongly_rejected']} | "
            f"{num(m['mean_confidence'], '{:.2f}')} / {num(h['mean_confidence'] if h else None, '{:.2f}')} | "
            f"{pct(m['high_confidence_accuracy'])} | {m['needs_review']} | {m['format_failures']} | {m['errors']} | "
            f"{bundle_cell(runs, mk, VISIBLE)} / {bundle_cell(runs, mk, HIDDEN)} | "
            f"{c['tokens_in_per_page']:,} / {c['tokens_out_per_page']:,} | {c['usd_per_1000_pages']:.2f} | "
            f"{m['latency_ms_p50'] / 1000:.1f} / {m['latency_ms_p95'] / 1000:.1f} |"
        )
    return lines


def best_picks(runs: dict, models: dict[str, Model]) -> list[str]:
    done = [mk for mk in models if metrics(runs, mk, VISIBLE)]
    if not done:
        return ["- (no runs yet)"]

    def label(mk):
        m = metrics(runs, mk, VISIBLE)
        return (f"**{models[mk].name}**: accuracy {pct(m['accuracy'])}, "
                f"${m['cost']['usd_per_1000_pages']:.2f} per 1,000 pages, p50 {m['latency_ms_p50'] / 1000:.1f} s")

    out = [f"- Best overall: {label(max(done, key=lambda mk: rank_key(runs, mk)))}"]
    perfect = [mk for mk in done if metrics(runs, mk, VISIBLE)["accuracy"] == 1.0]
    if perfect:
        out.append(f"- Cheapest at 100%: {label(min(perfect, key=lambda mk: metrics(runs, mk, VISIBLE)['cost']['usd_per_1000_pages']))}")
        out.append(f"- Fastest at 100%: {label(min(perfect, key=lambda mk: metrics(runs, mk, VISIBLE)['latency_ms_p50']))}")
    else:
        out.append("- No model reached 100% on the test images yet.")
    for mk in done:
        m = metrics(runs, mk, VISIBLE)
        if m["format_failures"] or m["errors"]:
            out.append(f"- Note: {models[mk].name} needed retries on {m['format_failures']} page(s) "
                       f"and failed on {m['errors']} page(s).")
    return out


def write_predictions(models: dict[str, Model], scenarios: list[str]) -> None:
    table: dict[str, dict] = {}
    columns = []
    for mk in models:
        for sn in scenarios:
            for which in ("images", "bundle"):
                path = RESULTS / mk / sn / f"{which}.json"
                if not path.exists():
                    continue
                col = f"{mk}:{sn}"
                if col not in columns:
                    columns += [col, f"{col}:conf"]
                for r in json.loads(path.read_text(encoding="utf-8")):
                    row = table.setdefault(r["id"], {"id": r["id"], "set": which, "expected": r["expected"]})
                    row[col] = r.get("label") or "ERROR"
                    if r.get("confidence") is not None:
                        row[f"{col}:conf"] = r["confidence"]
    with (RESULTS / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "set", "expected", *columns], restval="")
        writer.writeheader()
        writer.writerows(table.values())


def main() -> int:
    models, _ = load_models()
    scenarios = list(load_scenarios())
    runs = load_runs(models, scenarios)
    if not runs:
        print("No results yet; run: python -m scripts.eval.run_eval")
        return 1

    lines = ["# PlanSense — model comparison", "",
             "One model call per page. Test images: 24 pages (3 per class); bundle: 5-page PDF. "
             "Title hidden: the title-block corner is blanked out. Costs use the prices in models.json.", "",
             *model_table(runs, models), ""]
    missing = [mk for mk in models if not all((mk, sn) in runs for sn in scenarios)]
    if missing:
        lines += [f"Not run yet (or partly): {', '.join(models[m].name for m in missing)}. Run: "
                  f"`python -m scripts.eval.run_eval --model {' '.join(missing)}`", ""]
    lines += ["## Best picks", "", *best_picks(runs, models), "",
              "Per-page answers of every run: `predictions.csv`. Confusion matrices: `<model>/<scenario>/confusion_images.png`."]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_predictions(models, scenarios)
    print("\n".join(lines))
    print(f"\nWrote {RESULTS / 'report.md'} and {RESULTS / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
