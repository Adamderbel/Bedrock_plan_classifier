"""FastAPI service: POST /classify, GET /health, and the web UI at /.

Run:
    uvicorn app.app:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.schemas import PageResult
from app.pipeline.main import classify_file
from app.pipeline.render import CorruptFileError, TooManyPagesError, UnsupportedFileError, detect_kind
from app.pipeline.retrieval import IndexMissingError, get_index
from app.services.cost import step_prices

log = logging.getLogger("plansense")
WEB_DIR = Path(__file__).parent / "web"
CHUNK = 1024 * 1024

# Error names found in a page's `error` field -> HTTP status when every page failed.
THROTTLED = ("ThrottlingException", "TooManyRequests", "ServiceQuotaExceeded", "ServiceUnavailable")
TIMED_OUT = ("ReadTimeout", "ConnectTimeout", "ModelTimeoutException", "timed out")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load settings and the FAISS index once at startup, not on the first request.
    get_settings()
    try:
        get_index()
    except IndexMissingError as err:
        log.warning("%s (RAG requests will fail until it is built)", err)
    yield


app = FastAPI(title="PlanSense", version="1.0", lifespan=lifespan)


def _index_loaded() -> bool:
    try:
        get_index()
        return True
    except IndexMissingError:
        return False


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "index_loaded": _index_loaded(),
        "models": {"extract": s.sonnet_model_id, "decide": s.opus_model_id, "embed": s.embed_model_id},
        "defaults": {"use_rag": s.use_rag, "fast_path": s.fast_path_enabled},
        "limits": {"max_upload_mb": s.max_upload_mb, "max_pages": s.max_pages},
        "prices_usd_per_1m_tokens": {step: {"input": i, "output": o} for step, (i, o) in step_prices().items()},
    }


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read the upload, stopping as soon as it exceeds the size limit."""
    data = bytearray()
    while chunk := await upload.read(CHUNK):
        data += chunk
        if len(data) > max_bytes:
            raise HTTPException(413, f"File larger than {max_bytes // (1024 * 1024)} MB")
    return bytes(data)


def _all_failed_status(results: list[PageResult]) -> tuple[int, str] | None:
    """If every page failed, map the failure to an HTTP status; None otherwise."""
    if not results or any(r.error is None for r in results):
        return None
    errors = " ".join(r.error for r in results)
    if any(name in errors for name in THROTTLED):
        return 503, "Bedrock is throttling requests; retry shortly"
    if any(name in errors for name in TIMED_OUT):
        return 504, "Bedrock did not respond in time"
    if "IndexMissingError" in errors:
        return 503, "Knowledge base index not built; run: python -m scripts.ingest_kb"
    return 502, f"Model call failed: {results[0].error}"


@app.post("/classify")
async def classify(
    file: UploadFile = File(..., description="PDF, PNG or JPEG"),
    use_rag: bool | None = Query(None, description="Override USE_RAG for this request"),
    fast_path: bool | None = Query(None, description="Override FAST_PATH_ENABLED for this request"),
):
    s = get_settings()
    data = await _read_limited(file, s.max_upload_mb * 1024 * 1024)
    if not data:
        raise HTTPException(422, "Empty file")
    try:
        detect_kind(data)  # fast 415 before any rendering
    except UnsupportedFileError as err:
        raise HTTPException(415, str(err)) from err

    filename = Path(file.filename or "upload").name
    try:
        results = await run_in_threadpool(classify_file, data, filename, use_rag=use_rag, fast_path=fast_path)
    except UnsupportedFileError as err:
        raise HTTPException(415, str(err)) from err
    except (CorruptFileError, TooManyPagesError) as err:
        raise HTTPException(422, str(err)) from err

    failed = _all_failed_status(results)
    if failed:
        status, detail = failed
        headers = {"Retry-After": "30"} if status == 503 else None
        raise HTTPException(status, detail, headers=headers)
    return JSONResponse([r.model_dump(exclude_none=True) for r in results])


# The web UI; mounted last so it does not shadow the API routes.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
