import io
import os

import pytest
from PIL import Image

# Tests never call AWS, but settings require model IDs; set dummies when .env is absent.
os.environ.setdefault("SONNET_MODEL_ID", "test-extract-model")
os.environ.setdefault("OPUS_MODEL_ID", "test-decide-model")


def _image_bytes(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def make_image():
    """Factory: make_image(width, height, fmt="PNG", color="white") -> bytes."""

    def _make(width: int = 800, height: int = 600, fmt: str = "PNG", color="white") -> bytes:
        return _image_bytes(Image.new("RGB", (width, height), color), fmt)

    return _make


@pytest.fixture
def make_pdf():
    """Factory: make_pdf(pages) -> bytes of a PDF with `pages` blank pages."""

    def _make(pages: int = 1) -> bytes:
        imgs = [Image.new("RGB", (800, 600), "white") for _ in range(pages)]
        buf = io.BytesIO()
        imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:])
        return buf.getvalue()

    return _make
