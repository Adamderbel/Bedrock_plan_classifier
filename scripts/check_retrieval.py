"""Leave-one-out check of the knowledge base: does retrieval find the right neighbours?

Usage:
    python -m scripts.check_retrieval

Uses each KB example as a query (its vector is read back from the index, so no
AWS calls) and searches the other examples and the rules, the way a new page
would be searched at runtime. Reports per example and in total:
    top-1      the nearest other example has the same label
    vote       the majority label of the top-4 examples is correct
    agree      how many of the top-4 examples share the label (fast-path signal)
    rule rank  position of the correct discipline rule among the rules given to the
               decider (top-4, plus not_a_plan which is always included)
"""

from __future__ import annotations

import sys
from collections import Counter

from app.pipeline.retrieval import K_EXAMPLES, K_RULES, get_index, search, top_rules


def main() -> int:
    index = get_index()
    rows = []
    for i, doc_id in index.index_to_docstore_id.items():
        doc = index.docstore.search(doc_id)
        if doc.metadata["type"] != "example":
            continue
        vector = index.index.reconstruct(i).tolist()
        label, file = doc.metadata["label"], doc.metadata["file"]
        examples = search(vector, "example", K_EXAMPLES, exclude_file=file, index=index)
        rules = top_rules(vector, K_RULES, index=index)
        labels = [h.doc.metadata["label"] for h in examples]
        rule_ids = [h.doc.metadata["id"] for h in rules]
        rows.append(
            {
                "file": file,
                "label": label,
                "neighbours": labels,
                "top1": labels[0] == label,
                "vote": Counter(labels).most_common(1)[0][0] == label,
                "agree": labels.count(label),
                "rule_rank": rule_ids.index(label) + 1 if label in rule_ids else None,
                "rules": rule_ids,
            }
        )

    rows.sort(key=lambda r: r["file"])
    print(f"{'file':18} {'label':16} {'agree':5} {'rule':4}  neighbours (top-{K_EXAMPLES})  |  rules (top-{K_RULES})")
    for r in rows:
        mark = " " if r["vote"] else "x"
        rank = r["rule_rank"] or "-"
        print(
            f"{mark} {r['file']:16} {r['label']:16} {r['agree']}/{K_EXAMPLES}   {rank!s:4}  "
            f"{', '.join(r['neighbours'])}  |  {', '.join(r['rules'])}"
        )

    n = len(rows)
    pct = lambda count: f"{count}/{n} ({100 * count / n:.0f}%)"  # noqa: E731
    print()
    print(f"top-1 example correct : {pct(sum(r['top1'] for r in rows))}")
    print(f"top-4 majority correct: {pct(sum(r['vote'] for r in rows))}")
    print(f">=3 of 4 agree        : {pct(sum(r['agree'] >= 3 for r in rows))}")
    print(f"correct rule rank 1   : {pct(sum(r['rule_rank'] == 1 for r in rows))}")
    print(f"correct rule in context: {pct(sum(r['rule_rank'] is not None for r in rows))}")

    print("\nPer class (majority correct / >=3 agree):")
    for label in sorted({r["label"] for r in rows}):
        sub = [r for r in rows if r["label"] == label]
        print(
            f"  {label:16} {sum(r['vote'] for r in sub)}/{len(sub)}  "
            f"{sum(r['agree'] >= 3 for r in sub)}/{len(sub)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
