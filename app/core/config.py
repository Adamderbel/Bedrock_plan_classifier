"""Settings loaded from environment variables / .env (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required setting {name} (see .env.example)")
    return value


@dataclass(frozen=True)
class Settings:
    aws_region: str
    model_id: str

    review_threshold: float
    # Highest confidence allowed for a plan when no title block / sheet title is visible.
    no_title_confidence_cap: float

    max_upload_mb: int
    max_pages: int
    render_dpi: int
    max_image_side: int
    max_image_bytes: int
    page_workers: int

    # USD per 1M tokens, from the Bedrock pricing page for the region used.
    price_in: float
    price_out: float

    log_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            aws_region=os.getenv("AWS_REGION", "ap-south-1"),
            model_id=_required("MODEL_ID"),
            review_threshold=float(os.getenv("REVIEW_THRESHOLD", "0.6")),
            no_title_confidence_cap=float(os.getenv("NO_TITLE_CONFIDENCE_CAP", "0.75")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "20")),
            max_pages=int(os.getenv("MAX_PAGES", "20")),
            render_dpi=int(os.getenv("RENDER_DPI", "150")),
            # Vision models downsize anything much larger, so sending more only costs tokens.
            max_image_side=int(os.getenv("MAX_IMAGE_SIDE", "1568")),
            # Per image: PNG if it fits, else JPEG. Some models reject requests above ~2.5 MB.
            max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", "1000000")),
            # Pages of one PDF processed in parallel (bounded to limit throttling).
            page_workers=int(os.getenv("PAGE_WORKERS", "3")),
            # Defaults: Qwen3 VL 235B, on-demand, Asia Pacific (Mumbai).
            price_in=float(os.getenv("PRICE_IN", "0.62")),
            price_out=float(os.getenv("PRICE_OUT", "3.13")),
            log_dir=ROOT / os.getenv("LOG_DIR", "logs"),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
