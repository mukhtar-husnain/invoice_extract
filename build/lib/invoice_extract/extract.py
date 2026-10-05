"""Shared extraction prompt and Ollama / external vision API clients."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from PIL import Image

from invoice_extract.align import RawFieldExtraction
from invoice_extract.config import HEADER_FIELD_NAMES, LINE_ITEM_FIELD_NAMES, PipelineConfig
from invoice_extract.render import DocumentRender, PageRender, Token


EXTRACTION_SCHEMA_HINT = {
    "header": {name: {"value": "string|null", "tokenIds": [0], "confidence": 0.9} for name in HEADER_FIELD_NAMES},
    "lineItems": [
        {name: {"value": "string|null", "tokenIds": [0], "confidence": 0.9} for name in LINE_ITEM_FIELD_NAMES}
    ],
}


def build_token_catalog(tokens: list[Token]) -> list[dict[str, Any]]:
    return [
        {
            "id": t.id,
            "page": t.page,
            "text": t.text,
            "x": t.x,
            "y": t.y,
            "width": t.width,
            "height": t.height,
            "confidence": t.confidence,
        }
        for t in tokens
    ]


def build_prompt(document_id: str, tokens: list[Token]) -> str:
    catalog = build_token_catalog(tokens)
    return (
        "Extract invoice header fields and line items from the page images. "
        "Return ONLY valid JSON matching this shape:\n"
        f"{json.dumps(EXTRACTION_SCHEMA_HINT, indent=2)}\n\n"
        "Rules:\n"
        "- value must be printed text exactly as shown (whitespace collapsed).\n"
        "- tokenIds must reference ids from the token list below; do not invent coordinates.\n"
        "- Use null value and empty tokenIds when a field is absent.\n"
        "- confidence is 0.0-1.0 for your extraction certainty.\n\n"
        f"documentId: {document_id}\n"
        f"tokens: {json.dumps(catalog)}\n"
    )


def _image_to_base64(image: Image.Image, max_side: int) -> str:
    img = image.copy()
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_field(raw: dict[str, Any] | None) -> RawFieldExtraction:
    if not raw:
        return RawFieldExtraction(value=None, token_ids=[], confidence=None)
    value = raw.get("value")
    if value is not None:
        value = str(value)
    token_ids = [int(x) for x in raw.get("tokenIds") or []]
    conf = raw.get("confidence")
    confidence = float(conf) if conf is not None else None
    return RawFieldExtraction(value=value, token_ids=token_ids, confidence=confidence)


def parse_extraction_response(payload: dict[str, Any]) -> tuple[dict[str, RawFieldExtraction], list[dict[str, RawFieldExtraction]]]:
    header: dict[str, RawFieldExtraction] = {}
    for name in HEADER_FIELD_NAMES:
        header[name] = _parse_field(payload.get(name) if name in payload else payload.get("header", {}).get(name))

    line_items_raw = payload.get("lineItems") or []
    line_items: list[dict[str, RawFieldExtraction]] = []
    for row in line_items_raw:
        parsed_row: dict[str, RawFieldExtraction] = {}
        for name in LINE_ITEM_FIELD_NAMES:
            parsed_row[name] = _parse_field(row.get(name))
        line_items.append(parsed_row)
    return header, line_items


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, doc: DocumentRender) -> tuple[dict[str, RawFieldExtraction], list[dict[str, RawFieldExtraction]]]:
        raise NotImplementedError


class OllamaExtractor(BaseExtractor):
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def extract(self, doc: DocumentRender) -> tuple[dict[str, RawFieldExtraction], list[dict[str, RawFieldExtraction]]]:
        images = [_image_to_base64(p.image, self.config.image_max_side) for p in doc.pages]
        prompt = build_prompt(doc.document_id, doc.tokens)
        body = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "images": images,
        }
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{self.config.ollama_base_url}/api/generate", json=body)
            resp.raise_for_status()
            data = resp.json()
        payload = _extract_json(data.get("response", "{}"))
        return parse_extraction_response(payload)


class OpenAIExtractor(BaseExtractor):
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.api_key = os.environ["OPENAI_API_KEY"]

    def extract(self, doc: DocumentRender) -> tuple[dict[str, RawFieldExtraction], list[dict[str, RawFieldExtraction]]]:
        return _call_openai_compatible(
            doc,
            config=self.config,
            api_key=self.api_key,
            base_url="https://api.openai.com/v1",
            model=os.environ.get("INVOICE_OPENAI_MODEL", "gpt-4o"),
        )


class AnthropicExtractor(BaseExtractor):
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.api_key = os.environ["ANTHROPIC_API_KEY"]

    def extract(self, doc: DocumentRender) -> tuple[dict[str, RawFieldExtraction], list[dict[str, RawFieldExtraction]]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(doc.document_id, doc.tokens)}]
        for page in doc.pages:
            b64 = _image_to_base64(page.image, self.config.image_max_side)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                }
            )
        body = {
            "model": os.environ.get("INVOICE_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        with httpx.Client(timeout=300.0) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        text = "".join(part.get("text", "") for part in data.get("content", []) if part.get("type") == "text")
        return parse_extraction_response(_extract_json(text))


class GeminiExtractor(BaseExtractor):
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.api_key = os.environ["GEMINI_API_KEY"]

    def extract(self, doc: DocumentRender) -> tuple[dict[str, RawFieldExtraction], list[dict[str, RawFieldExtraction]]]:
        parts: list[dict[str, Any]] = [{"text": build_prompt(doc.document_id, doc.tokens)}]
        for page in doc.pages:
            b64 = _image_to_base64(page.image, self.config.image_max_side)
            parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
        model = os.environ.get("INVOICE_GEMINI_MODEL", "gemini-2.0-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {"contents": [{"parts": parts}]}
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, params={"key": self.api_key}, json=body)
            resp.raise_for_status()
            data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return parse_extraction_response(_extract_json(text))


def _call_openai_compatible(
    doc: DocumentRender,
    *,
    config: PipelineConfig,
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[dict[str, RawFieldExtraction], list[dict[str, RawFieldExtraction]]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": build_prompt(doc.document_id, doc.tokens)}]
    for page in doc.pages:
        b64 = _image_to_base64(page.image, config.image_max_side)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    body = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(f"{base_url}/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return parse_extraction_response(_extract_json(text))


def ollama_is_available(config: PipelineConfig) -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{config.ollama_base_url}/api/tags")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


def api_key_is_available(config: PipelineConfig) -> bool:
    provider = config.llm_provider
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def build_api_extractor(config: PipelineConfig) -> BaseExtractor:
    provider = config.llm_provider
    if provider == "anthropic":
        return AnthropicExtractor(config)
    if provider == "gemini":
        return GeminiExtractor(config)
    return OpenAIExtractor(config)


def default_page_size(pages: list[PageRender]) -> tuple[int, int]:
    if not pages:
        return 1, 1
    return pages[0].width, pages[0].height
