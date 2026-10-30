import io
import os

import pytest
from PIL import Image

from app.pipeline.render import (
    CorruptFileError,
    TooManyPagesError,
    UnsupportedFileError,
    detect_kind,
    encode_image,
    render_pages,
)


def _decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def test_detect_kind(make_image, make_pdf):
    assert detect_kind(make_image(fmt="PNG")) == "png"
    assert detect_kind(make_image(fmt="JPEG")) == "jpeg"
    assert detect_kind(make_pdf()) == "pdf"


@pytest.mark.parametrize("data", [b"GIF89a....", b"hello world", b""])
def test_unsupported_file(data):
    with pytest.raises(UnsupportedFileError):
        detect_kind(data)


def test_png_gives_one_page(make_image):
    pages = render_pages(make_image(800, 600))
    assert len(pages) == 1
    assert pages[0].page == 1
    assert pages[0].media_type == "image/png"
    assert pages[0].size == (800, 600)


def test_large_image_is_downscaled_keeping_aspect(make_image):
    (page,) = render_pages(make_image(4000, 3000, fmt="JPEG"), max_side=1568)
    assert page.size == (1568, 1176)
    assert _decode(page.image).size == (1568, 1176)


def test_multipage_pdf_gives_one_result_per_page(make_pdf):
    pages = render_pages(make_pdf(3))
    assert [p.page for p in pages] == [1, 2, 3]


def test_too_many_pages(make_pdf):
    with pytest.raises(TooManyPagesError):
        render_pages(make_pdf(3), max_pages=2)


def test_corrupt_pdf():
    with pytest.raises(CorruptFileError):
        render_pages(b"%PDF-1.4\nthis is not really a pdf")


def test_truncated_png(make_image):
    with pytest.raises(CorruptFileError):
        render_pages(make_image()[:40])


def test_title_block_crop_is_bottom_right():
    img = Image.new("RGB", (1000, 1000), "white")
    img.paste((255, 0, 0), (900, 900, 1000, 1000))  # red square bottom-right
    img.paste((0, 0, 255), (0, 0, 100, 100))  # blue square top-left
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    (page,) = render_pages(buf.getvalue())
    crop = _decode(page.title_block)
    colors = {c for _, c in crop.getcolors(maxcolors=10_000)}
    assert (255, 0, 0) in colors
    assert (0, 0, 255) not in colors


def test_rgba_input_is_accepted():
    buf = io.BytesIO()
    Image.new("RGBA", (200, 100), (0, 0, 0, 0)).save(buf, format="PNG")
    (page,) = render_pages(buf.getvalue())
    assert _decode(page.image).mode == "RGB"


def test_encode_falls_back_to_jpeg_when_too_big():
    data, media_type = encode_image(Image.new("RGB", (100, 100), "white"), max_bytes=10)
    assert media_type == "image/jpeg"
    assert data.startswith(b"\xff\xd8\xff")


def _noise(size: int) -> Image.Image:
    return Image.frombytes("RGB", (size, size), os.urandom(size * size * 3))


def test_encode_keeps_small_images_as_png():
    data, media_type = encode_image(Image.new("RGB", (1568, 1100), "white"))
    assert media_type == "image/png" and len(data) <= 1_000_000


def test_encode_fits_large_images_in_the_budget():
    # Random noise barely compresses: PNG ~3 MB, so it needs lower JPEG quality or shrinking.
    data, media_type = encode_image(_noise(1000), max_bytes=300_000)
    assert media_type == "image/jpeg" and len(data) <= 300_000


def test_render_respects_the_byte_budget():
    buf = io.BytesIO()
    _noise(1200).save(buf, format="PNG")
    (page,) = render_pages(buf.getvalue(), max_image_bytes=400_000)
    assert len(page.image) <= 400_000 and len(page.title_block) <= 400_000
