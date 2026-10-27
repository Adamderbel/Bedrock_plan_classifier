"""Bedrock clients: extractor and decider chat models, Titan embeddings."""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from typing import TypeVar

from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.schemas import Usage

T = TypeVar("T", bound=BaseModel)

# Adaptive retries back off automatically on ThrottlingException / 5xx.
BOTO_CONFIG = Config(
    connect_timeout=10,
    read_timeout=120,
    retries={"mode": "adaptive", "max_attempts": 8},
)


def make_chat(model_id: str, max_tokens: int = 2048) -> ChatBedrockConverse:
    s = get_settings()
    extra = None
    if "amazon.nova" in model_id:
        # Nova still samples at temperature 0; greedy decoding (topK=1) is what
        # AWS recommends to avoid malformed tool calls.
        extra = {"inferenceConfig": {"topK": 1}}
    return ChatBedrockConverse(
        model=model_id,
        region_name=s.aws_region,
        temperature=0,
        max_tokens=max_tokens,
        additional_model_request_fields=extra,
        config=BOTO_CONFIG,
    )


def message_text(message) -> str:
    """Plain text of a chat response (content may be a string or a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


@lru_cache
def get_embeddings() -> BedrockEmbeddings:
    s = get_settings()
    # Normalised vectors make FAISS's L2 ranking equivalent to cosine similarity.
    return BedrockEmbeddings(
        model_id=s.embed_model_id,
        region_name=s.embed_region,
        normalize=True,
        config=BOTO_CONFIG,
    )


def image_block(data: bytes, media_type: str) -> dict:
    """LangChain content block for an image, sent to Bedrock as raw bytes."""
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}


def usage_from(raw_message) -> Usage:
    meta = getattr(raw_message, "usage_metadata", None) or {}
    return Usage(input_tokens=meta.get("input_tokens", 0), output_tokens=meta.get("output_tokens", 0))


class StructuredOutputError(RuntimeError):
    """The model did not return output matching the schema, even after retries."""


@lru_cache
def _chat(model_id: str, max_tokens: int) -> ChatBedrockConverse:
    return make_chat(model_id, max_tokens=max_tokens)


@lru_cache
def _structured(model_id: str, schema: type[BaseModel], max_tokens: int):
    return _chat(model_id, max_tokens).with_structured_output(schema, include_raw=True)


def _is_model_error(err: Exception) -> bool:
    """Bedrock rejects malformed tool calls with ModelErrorException; worth retrying."""
    return isinstance(err, ClientError) and err.response.get("Error", {}).get("Code") == "ModelErrorException"


def _parse_json(text: str, schema: type[T]) -> T:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the response")
    return schema.model_validate_json(text[start : end + 1])


def invoke_structured(
    model_id: str,
    schema: type[T],
    system: str,
    content: list[dict],
    instruction: str,
    *,
    max_tokens: int = 2048,
    attempts: int = 2,
) -> tuple[T, Usage]:
    """Call a chat model and return (parsed schema object, summed token usage).

    Tries structured output (tool use) up to `attempts` times, retrying malformed
    tool calls and schema mismatches, then falls back once to asking for plain
    JSON and validating it with Pydantic.
    """
    usage = Usage()

    def add_usage(raw) -> None:
        call = usage_from(raw)
        usage.input_tokens += call.input_tokens
        usage.output_tokens += call.output_tokens

    errors: list[str] = []
    messages = [SystemMessage(system), HumanMessage(content=content + [{"type": "text", "text": instruction}])]
    for _ in range(attempts):
        try:
            out = _structured(model_id, schema, max_tokens).invoke(messages)
        except ClientError as err:
            if not _is_model_error(err):
                raise
            errors.append("malformed tool call")
            continue
        add_usage(out["raw"])
        if out["parsed"] is not None:
            return out["parsed"], usage
        errors.append(f"schema mismatch: {out.get('parsing_error')}")

    # Fallback: no tool use, plain JSON validated with Pydantic.
    fallback = (
        f"{instruction}\nReturn a single JSON object, with no other text, matching this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    messages = [SystemMessage(system), HumanMessage(content=content + [{"type": "text", "text": fallback}])]
    raw = _chat(model_id, max_tokens).invoke(messages)
    add_usage(raw)
    try:
        return _parse_json(message_text(raw), schema), usage
    except (ValueError, ValidationError) as err:
        errors.append(f"JSON fallback: {err}")
    raise StructuredOutputError(f"{schema.__name__} not returned: " + "; ".join(errors))
