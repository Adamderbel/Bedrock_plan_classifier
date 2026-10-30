"""Turn an uploaded PDF / PNG / JPEG into page images ready for the vision model."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

FileKind = Literal["pdf", "png", "jpeg"]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
PDF_MAGIC = b"%PDF-"

# Bytes per image. Bedrock accepts up to 3.75 MB per image, but some models
# (Mistral, Kimi, Qwen, Nemotron) reject whole requests above ~2.5 MB, and every
# call sends two images (full page + title-block crop).
MAX_IMAGE_BYTES = 1_000_000
JPEG_QUALITIES = (85, 75, 65, 55)

# Title blocks sit in the bottom-right corner on most sheets.
TITLE_BLOCK_BOX = (0.6, 0.6, 1.0, 1.0)  # left, top, right, bottom as fractions


class RenderError(ValueError):
    """Base class for files we cannot turn into images."""


class UnsupportedFileError(RenderError):
    pass


class CorruptFileError(RenderError):
    pass


class TooManyPagesError(RenderError):
    pass


@dataclass(frozen=True)
class PageImage:
    page: int  # 1-based
    image: bytes
    media_type: str
    title_block: bytes
    title_block_media_type: str
    size: tuple[int, int]  # (width, height) of `image`


def detect_kind(data: bytes) -> FileKind:
    """Identify the file from its magic bytes, not its extension."""
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    # The PDF spec tolerates junk before the header within the first 1 KB.
    if PDF_MAGIC in data[:1024]:
        return "pdf"
    raise UnsupportedFileError("Unsupported file type: expected PDF, PNG or JPEG")


def render_pages(
    data: bytes,
    *,
    dpi: int = 150,
    max_pages: int = 20,
    max_side: int = 1568,
    hide_title_block: bool = False,
    max_image_bytes: int = MAX_IMAGE_BYTES,
) -> list[PageImage]:
    """Render a PDF / PNG / JPEG into pages.

    Each image is at most max_image_bytes (see encode_image).
    hide_title_block paints the title-block corner white (evaluation only: it
    simulates pages whose title block gives no answer).
    """
    kind = detect_kind(data)
    images = _pdf_to_images(data, dpi, max_pages) if kind == "pdf" else [_open_image(data)]
    if hide_title_block:
        images = [_blank_title_block(img) for img in images]
    return [_to_page_image(i + 1, img, max_side, max_image_bytes) for i, img in enumerate(images)]


def _blank_title_block(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    left, top, right, bottom = TITLE_BLOCK_BOX
    ImageDraw.Draw(img).rectangle((int(w * left), int(h * top), int(w * right), int(h * bottom)), fill="white")
    return img


def crop_title_block(img: Image.Image) -> Image.Image:
    w, h = img.size
    left, top, right, bottom = TITLE_BLOCK_BOX
    return img.crop((int(w * left), int(h * top), int(w * right), int(h * bottom)))


def downscale(img: Image.Image, max_side: int) -> Image.Image:
    """Shrink so the long side is <= max_side; never upscale."""
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def encode_image(img: Image.Image, max_bytes: int = MAX_IMAGE_BYTES) -> tuple[bytes, str]:
    """PNG keeps line drawings crisp, so it is used when it fits in max_bytes.

    Otherwise JPEG, lowering the quality step by step and, as a last resort,
    shrinking the image until it fits.
    """
    data = _save(img, "PNG", optimize=True)
    if len(data) <= max_bytes:
        return data, "image/png"
    while True:
        for quality in JPEG_QUALITIES:
            data = _save(img, "JPEG", quality=quality)
            if len(data) <= max_bytes:
                return data, "image/jpeg"
        if max(img.size) <= 256:
            return data, "image/jpeg"
        img = downscale(img, int(max(img.size) * 0.75))


def _save(img: Image.Image, fmt: str, **options) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **options)
    return buf.getvalue()


def _pdf_to_images(data: bytes, dpi: int, max_pages: int) -> list[Image.Image]:
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as err:
        raise CorruptFileError("Cannot open PDF (corrupt or password-protected)") from err
    try:
        n = len(pdf)
        if n == 0:
            raise CorruptFileError("PDF has no pages")
        if n > max_pages:
            raise TooManyPagesError(f"PDF has {n} pages; the limit is {max_pages}")
        return [pdf[i].render(scale=dpi / 72).to_pil() for i in range(n)]
    finally:
        pdf.close()


def _open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
        raise CorruptFileError("Cannot decode image (corrupt or too large)") from err
    # Phone photos often store rotation in EXIF instead of the pixels.
    return ImageOps.exif_transpose(img)


def _to_page_image(page: int, img: Image.Image, max_side: int, max_bytes: int = MAX_IMAGE_BYTES) -> PageImage:
    img = img.convert("RGB")
    # Crop from the full-resolution render so small title-block text stays legible.
    crop = downscale(crop_title_block(img), max_side)
    full = downscale(img, max_side)
    image, media_type = encode_image(full, max_bytes)
    title_block, tb_media_type = encode_image(crop, max_bytes)
    return PageImage(page, image, media_type, title_block, tb_media_type, full.size)
