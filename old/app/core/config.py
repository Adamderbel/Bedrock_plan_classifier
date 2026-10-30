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


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    aws_region: str
    embed_region: str
    sonnet_model_id: str
    opus_model_id: str
    embed_model_id: str

    use_rag: bool
    fast_path_enabled: bool
    fast_path_min_agreement: int
    review_threshold: float

    max_upload_mb: int
    max_pages: int
    render_dpi: int
    max_image_side: int
    page_workers: int

    # USD per 1M tokens, from the Bedrock pricing page for the region used.
    price_extract_in: float
    price_extract_out: float
    price_decide_in: float
    price_decide_out: float
    price_embed_in: float

    kb_index_dir: Path
    log_dir: Path

    @classmethod
    def from_env(cls) -> Settings:
        region = os.getenv("AWS_REGION", "eu-north-1")
        return cls(
            aws_region=region,
            embed_region=os.getenv("EMBED_REGION", region),
            sonnet_model_id=_required("SONNET_MODEL_ID"),
            opus_model_id=_required("OPUS_MODEL_ID"),
            embed_model_id=os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            use_rag=_bool("USE_RAG", True),
            fast_path_enabled=_bool("FAST_PATH_ENABLED", False),
            fast_path_min_agreement=int(os.getenv("FAST_PATH_MIN_AGREEMENT", "3")),
            review_threshold=float(os.getenv("REVIEW_THRESHOLD", "0.6")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "20")),
            max_pages=int(os.getenv("MAX_PAGES", "20")),
            render_dpi=int(os.getenv("RENDER_DPI", "150")),
            # Claude downsizes anything larger than ~1568 px on the long side,
            # so sending more only costs tokens.
            max_image_side=int(os.getenv("MAX_IMAGE_SIDE", "1568")),
            # Pages of one PDF processed in parallel (bounded to limit throttling).
            page_workers=int(os.getenv("PAGE_WORKERS", "3")),
            # Defaults: Amazon Nova Pro (extract + decide) and Titan Text Embeddings V2.
            price_extract_in=float(os.getenv("PRICE_EXTRACT_IN", "0.80")),
            price_extract_out=float(os.getenv("PRICE_EXTRACT_OUT", "3.20")),
            price_decide_in=float(os.getenv("PRICE_DECIDE_IN", "0.80")),
            price_decide_out=float(os.getenv("PRICE_DECIDE_OUT", "3.20")),
            price_embed_in=float(os.getenv("PRICE_EMBED_IN", "0.02")),
            kb_index_dir=ROOT / os.getenv("KB_INDEX_DIR", "kb_index"),
            log_dir=ROOT / os.getenv("LOG_DIR", "logs"),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
