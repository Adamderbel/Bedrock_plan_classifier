"""Smoke test: verify Bedrock access for the extractor, decider and Titan models.

Usage:
    python -m scripts.check_bedrock

Reads settings from .env (see .env.example). For each chat model it makes a
plain text call, then an image + tool call (the pipeline's structured output
relies on tool use). Titan gets one embedding call. Cost well under $0.01.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import boto3
import botocore
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

REGION = os.getenv("AWS_REGION", "eu-north-1")
EMBED_REGION = os.getenv("EMBED_REGION", REGION)
SONNET_ID = os.getenv("SONNET_MODEL_ID", "")
OPUS_ID = os.getenv("OPUS_MODEL_ID", "")
EMBED_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

# Fallback regions tried for Titan if EMBED_REGION does not serve it.
EMBED_FALLBACK_REGIONS = ["ap-south-1", "eu-north-1", "eu-west-1", "eu-central-1"]

BOTO_CONFIG = Config(
    connect_timeout=10,
    read_timeout=60,
    retries={"mode": "standard", "max_attempts": 3},
)

HINTS = {
    "AccessDeniedException": "No permission, the model is not enabled yet, or its Marketplace "
    "subscription failed (e.g. INVALID_PAYMENT_INSTRUMENT).",
    "UnrecognizedClientException": "The API key is invalid or expired: generate a new short-term key.",
    "ExpiredTokenException": "The API key expired: generate a new short-term key.",
    "ValidationException": "Wrong model ID, model not in this region, or the model does not "
    "support this input (images / tools).",
    "ResourceNotFoundException": "Model/profile not found in this region: check the ID and region.",
    "ThrottlingException": "Rate limited: wait a few seconds and retry.",
}


def client(region: str):
    return boto3.client("bedrock-runtime", region_name=region, config=BOTO_CONFIG)


def explain(err: Exception) -> str:
    if isinstance(err, ClientError):
        code = err.response.get("Error", {}).get("Code", "Unknown")
        msg = err.response.get("Error", {}).get("Message", "")
        return f"{code}: {msg}\n      hint: {HINTS.get(code, 'see the message above')}"
    return f"{type(err).__name__}: {err}"


# A tiny tool the model must call; mirrors how with_structured_output works.
COLOR_TOOL = {
    "tools": [
        {
            "toolSpec": {
                "name": "report_color",
                "description": "Report the main color of the image.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"color": {"type": "string"}},
                        "required": ["color"],
                    }
                },
            }
        }
    ]
}


def red_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def check_text(name: str, model_id: str) -> bool:
    if not model_id:
        print(f"[FAIL] {name}: model ID missing in .env")
        return False
    start = time.perf_counter()
    try:
        resp = client(REGION).converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Reply with exactly: OK"}]}],
            inferenceConfig={"maxTokens": 10},
        )
    except (ClientError, BotoCoreError) as err:
        print(f"[FAIL] {name} text ({model_id} @ {REGION})\n      {explain(err)}")
        return False
    ms = (time.perf_counter() - start) * 1000
    text = resp["output"]["message"]["content"][0].get("text", "").strip()
    usage = resp.get("usage", {})
    print(
        f"[ OK ] {name} text ({model_id} @ {REGION}) -> {text!r} "
        f"| tokens in/out {usage.get('inputTokens')}/{usage.get('outputTokens')} | {ms:.0f} ms"
    )
    return True


def check_image_tool(name: str, model_id: str) -> bool:
    """Send an image and require a tool call back."""
    try:
        resp = client(REGION).converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": red_png()}}},
                        {"text": "Call the report_color tool with the main color of this image."},
                    ],
                }
            ],
            toolConfig=COLOR_TOOL,
            inferenceConfig={"maxTokens": 200, "temperature": 0},
        )
    except (ClientError, BotoCoreError) as err:
        print(f"[FAIL] {name} image+tool ({model_id})\n      {explain(err)}")
        return False
    content = resp["output"]["message"]["content"]
    calls = [c["toolUse"] for c in content if "toolUse" in c]
    if not calls:
        text = " ".join(c.get("text", "") for c in content).strip()[:120]
        print(f"[FAIL] {name} image+tool: answered without calling the tool -> {text!r}")
        return False
    print(f"[ OK ] {name} image+tool -> {calls[0]['name']}({json.dumps(calls[0]['input'])})")
    return True


def check_chat(name: str, model_id: str) -> bool:
    return check_text(name, model_id) and check_image_tool(name, model_id)


def check_titan() -> bool:
    regions = [EMBED_REGION] + [r for r in EMBED_FALLBACK_REGIONS if r != EMBED_REGION]
    body = json.dumps({"inputText": "floor plan with doors and windows", "dimensions": 1024, "normalize": True})
    for region in regions:
        try:
            resp = client(region).invoke_model(
                modelId=EMBED_ID, body=body, contentType="application/json", accept="application/json"
            )
        except (ClientError, BotoCoreError) as err:
            print(f"[FAIL] Titan ({EMBED_ID} @ {region})\n      {explain(err)}")
            continue
        vec = json.loads(resp["body"].read())["embedding"]
        print(f"[ OK ] Titan ({EMBED_ID} @ {region}) -> vector of {len(vec)} dims")
        if region != EMBED_REGION:
            print(f"      note: set EMBED_REGION={region} in .env")
        return True
    return False


def main() -> int:
    print(f"boto3 {boto3.__version__} / botocore {botocore.__version__}")
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        print("auth: Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)")
    else:
        print("auth: no API key in .env -> falling back to the default AWS credential chain")
    print()

    results = [
        check_chat("Extractor", SONNET_ID),
        check_chat("Decider", OPUS_ID),
        check_titan(),
    ]
    print()
    if all(results):
        print("All checks passed.")
        return 0
    print(f"{results.count(False)} check(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
