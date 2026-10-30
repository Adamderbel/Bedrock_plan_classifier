"""Build data/kb and data/test from data/manifest.csv.

Usage:
    python -m scripts.download_data            # download + render everything missing
    python -m scripts.download_data --force    # rebuild every file

Manifest columns:
    id      file stem, e.g. ex_elec_04
    split   kb | test
    label   one of the 8 labels
    url     http(s) URL, or local:<path relative to repo> for files you add by hand
    page    1-based page to extract from a PDF (blank for images)
    source  human-readable origin (project / site)
    license license or terms of the source

Output:
    kb   -> data/kb/<label>/<id>.png          (page rendered at 150 DPI)
    test -> data/test/<label>/<id>.pdf|.png|.jpg  (PDF pages kept as single-page PDFs)
    data/test_multipage/bundle.pdf + bundle.json  (several test files merged into one
            PDF, with the expected label per page, to check one-result-per-page)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from app.core.schemas import LABELS
from app.pipeline.render import detect_kind

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
MANIFEST = DATA / "manifest.csv"

MULTIPAGE_DIR = DATA / "test_multipage"
# Test files merged into one PDF, in page order: different disciplines plus one non-plan.
BUNDLE_PAGES = ["test_elec_01", "test_plumb_01", "test_hvac_02", "test_civil_03", "test_nap_02"]

USER_AGENT = "PlanSense-dataset/0.1 (technical exercise; contact: repo owner)"
TARGET_PER_LABEL = {"kb": 5, "test": 3}
RENDER_DPI = 150
MAX_SIDE = 5000  # keep huge sheets manageable on disk; pipeline downsizes further

Image.MAX_IMAGE_PIXELS = 200_000_000  # large drawing sheets are legit


def fetch(url: str) -> bytes:
    """Download once into data/raw (keyed by URL hash) and reuse afterwards."""
    if url.startswith("local:"):
        return (ROOT / url.removeprefix("local:")).read_bytes()
    RAW.mkdir(parents=True, exist_ok=True)
    cached = RAW / hashlib.sha1(url.encode()).hexdigest()[:16]
    if cached.exists():
        return cached.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    cached.write_bytes(data)
    time.sleep(1)  # be polite to public servers
    return data


def render_pdf_page(data: bytes, page: int, dpi: int = RENDER_DPI) -> Image.Image:
    pdf = pdfium.PdfDocument(data)
    try:
        img = pdf[page - 1].render(scale=dpi / 72).to_pil()
    finally:
        pdf.close()
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return img.convert("RGB")


def extract_pdf_page(data: bytes, page: int) -> bytes:
    src = pdfium.PdfDocument(data)
    dst = pdfium.PdfDocument.new()
    try:
        dst.import_pages(src, [page - 1])
        buf = io.BytesIO()
        dst.save(buf)
        return buf.getvalue()
    finally:
        dst.close()
        src.close()


def build_row(row: dict[str, str], force: bool) -> Path:
    split, label, stem = row["split"], row["label"], row["id"]
    if split not in TARGET_PER_LABEL:
        raise ValueError(f"bad split {split!r}")
    if label not in LABELS:
        raise ValueError(f"bad label {label!r}")
    out_dir = DATA / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob(f"{stem}.*"))
    if existing and not force:
        return existing[0]

    data = fetch(row["url"])
    kind = detect_kind(data)
    page = int(row["page"]) if row.get("page") else 1

    if split == "test" and kind == "pdf":
        out = out_dir / f"{stem}.pdf"
        out.write_bytes(extract_pdf_page(data, page))
    elif split == "test":
        out = out_dir / f"{stem}.{'jpg' if kind == 'jpeg' else 'png'}"
        out.write_bytes(data)
    else:
        img = render_pdf_page(data, page) if kind == "pdf" else Image.open(io.BytesIO(data)).convert("RGB")
        out = out_dir / f"{stem}.png"
        img.save(out, format="PNG", optimize=True)
    return out


def build_bundle(rows: list[dict[str, str]], force: bool) -> Path:
    """Merge BUNDLE_PAGES (already built test files) into one multi-page PDF."""
    out = MULTIPAGE_DIR / "bundle.pdf"
    if out.exists() and not force:
        return out
    by_id = {r["id"]: r for r in rows}
    images, expected = [], []
    for page, stem in enumerate(BUNDLE_PAGES, start=1):
        row = by_id[stem]
        (src,) = (DATA / "test" / row["label"]).glob(f"{stem}.*")
        images.append(Image.open(src).convert("RGB"))
        expected.append({"page": page, "id": stem, "label": row["label"]})
    MULTIPAGE_DIR.mkdir(parents=True, exist_ok=True)
    # resolution=150 so re-rendering the PDF at 150 DPI gives back the original pixels.
    images[0].save(out, format="PDF", save_all=True, append_images=images[1:], resolution=RENDER_DPI)
    (MULTIPAGE_DIR / "bundle.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild files that already exist")
    args = parser.parse_args()

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("id") and not r["id"].startswith("#")]

    counts: Counter[tuple[str, str]] = Counter()
    failures = 0
    for row in rows:
        try:
            out = build_row(row, args.force)
            counts[(row["split"], row["label"])] += 1
            print(f"[ OK ] {row['split']:4} {row['label']:16} {out.relative_to(ROOT)}")
        except Exception as err:  # keep going; report at the end
            failures += 1
            print(f"[FAIL] {row['split']:4} {row['label']:16} {row['id']}: {type(err).__name__}: {err}")

    try:
        bundle = build_bundle(rows, args.force)
        print(f"[ OK ] multipage bundle ({len(BUNDLE_PAGES)} pages) {bundle.relative_to(ROOT)}")
    except Exception as err:
        failures += 1
        print(f"[FAIL] multipage bundle: {type(err).__name__}: {err}")

    print("\nFiles per class (have / target):")
    for split, target in TARGET_PER_LABEL.items():
        line = "  ".join(f"{label}={counts[(split, label)]}/{target}" for label in LABELS)
        print(f"  {split:4} {line}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
