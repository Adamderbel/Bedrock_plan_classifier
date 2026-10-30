"""Step 1 - build the FAISS knowledge base (kb_index/).

Usage:
    python -m scripts.ingest_kb             # reuse cached Sonnet extractions
    python -m scripts.ingest_kb --refresh   # re-extract every example with Sonnet

Indexes two kinds of documents with Titan text embeddings:
    rule     one per "## <label>" section of data/kb/disciplines.md
             (or one per "### <part>" when a section is split)
    example  one per image in data/kb/<label>/, using Sonnet's extracted features
             (cached in data/kb_features/<id>.json so re-indexing costs nothing)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import ROOT, get_settings
from app.core.schemas import LABELS, PlanFeatures, Usage
from app.pipeline.extract import extract_features
from app.pipeline.llm import get_embeddings
from app.pipeline.render import render_pages

KB_DIR = ROOT / "data" / "kb"
RULES_FILE = KB_DIR / "disciplines.md"
FEATURES_DIR = ROOT / "data" / "kb_features"
MANIFEST = ROOT / "data" / "manifest.csv"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
WORKERS = 4  # parallel Sonnet calls; adaptive retries absorb throttling


def load_rules() -> list[Document]:
    """Split disciplines.md into rule documents.

    One document per '## <label>' section. A section with '### <part>' headings
    is split into one document per part, each prefixed with the section intro,
    so each vector stays focused on one kind of drawing.
    """
    sections: dict[str, list[str]] = {}
    current = None
    for line in RULES_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            if current not in LABELS:
                raise ValueError(f"Unknown label heading in {RULES_FILE.name}: {current!r}")
            sections[current] = [line]
        elif current:
            sections[current].append(line)
    missing = set(LABELS) - sections.keys()
    if missing:
        raise ValueError(f"{RULES_FILE.name} has no section for: {sorted(missing)}")

    source = RULES_FILE.relative_to(ROOT).as_posix()
    docs = []
    for label, lines in sections.items():
        intro, parts, part = [], {}, None
        for line in lines:
            if line.startswith("### "):
                part = line[4:].strip()
                parts[part] = [line]
            elif part:
                parts[part].append(line)
            else:
                intro.append(line)
        if not parts:
            parts = {None: []}
        for name, body in parts.items():
            metadata = {"type": "rule", "id": label, "source": source}
            if name:
                metadata["part"] = name
            docs.append(Document(page_content="\n".join(intro + body).strip(), metadata=metadata))
    return docs


def load_sources() -> dict[str, str]:
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return {r["id"]: r["source"] for r in csv.DictReader(f)}


def example_files() -> list[tuple[str, Path]]:
    return [
        (label, path)
        for label in LABELS
        for path in sorted((KB_DIR / label).iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]


def get_features(path: Path, refresh: bool) -> tuple[PlanFeatures, Usage, bool]:
    """Return (features, usage, cached?) for one example image."""
    cache = FEATURES_DIR / f"{path.stem}.json"
    if cache.exists() and not refresh:
        return PlanFeatures.model_validate_json(cache.read_text(encoding="utf-8")), Usage(), True
    s = get_settings()
    (page,) = render_pages(path.read_bytes(), dpi=s.render_dpi, max_pages=1, max_side=s.max_image_side)
    features, usage = extract_features(page)
    cache.write_text(features.model_dump_json(indent=2), encoding="utf-8")
    return features, usage, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="ignore cached extractions and call the extractor again")
    args = parser.parse_args()

    settings = get_settings()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    rules = load_rules()
    print(f"Rules: {len(rules)} documents for {len(LABELS)} labels from {RULES_FILE.relative_to(ROOT)}")

    sources = load_sources()
    files = example_files()
    print(f"Examples: {len(files)} images, extracting with {settings.sonnet_model_id}\n")

    examples: list[Document] = []
    total = Usage()
    failures = 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(get_features, path, args.refresh): (label, path) for label, path in files}
        for future in as_completed(futures):
            label, path = futures[future]
            try:
                features, usage, cached = future.result()
            except Exception as err:  # keep indexing the rest; report at the end
                failures += 1
                print(f"[FAIL] {label:16} {path.name}: {type(err).__name__}: {err}")
                continue
            total.input_tokens += usage.input_tokens
            total.output_tokens += usage.output_tokens
            if features.preliminary_label != label:
                note = f"  (extractor guessed {features.preliminary_label})"
            else:
                note = ""
            print(f"[{'CACHE' if cached else ' OK  '}] {label:16} {path.name}{note}")
            examples.append(
                Document(
                    page_content=features.to_embedding_text(),
                    metadata={
                        "type": "example",
                        "label": label,
                        "file": path.name,
                        "source": sources.get(path.stem, ""),
                        "sheet_number": features.sheet_number,
                    },
                )
            )

    print(
        f"\nExtractor: {total.input_tokens} input / {total.output_tokens} output tokens "
        f"in {time.perf_counter() - start:.0f}s"
    )

    # Stable order so rebuilding the index gives the same result.
    examples.sort(key=lambda d: d.metadata["file"])
    docs = rules + examples
    print(f"Embedding {len(docs)} documents with {settings.embed_model_id} ...")
    index = FAISS.from_documents(docs, get_embeddings())
    index.save_local(str(settings.kb_index_dir))
    print(f"Saved index to {settings.kb_index_dir.relative_to(ROOT)} ({len(rules)} rules + {len(examples)} examples)")

    if failures:
        print(f"\n{failures} example(s) failed and are missing from the index; rerun to retry them.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
