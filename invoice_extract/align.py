"""Align extracted values to cited token boxes."""

from __future__ import annotations

import re
from dataclasses import dataclass

from invoice_extract.render import Token, union_bbox
from invoice_extract.schema import BBox, FieldValue


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


@dataclass
class RawFieldExtraction:
    value: str | None
    token_ids: list[int]
    confidence: float | None = None
    normalized_bbox: tuple[float, float, float, float] | None = None
    page: int | None = None


@dataclass
class AlignedField:
    field: FieldValue
    mean_ocr_confidence: float | None
    model_confidence: float | None
    value_in_tokens: bool
    token_grounded: bool
    spans_pages: bool = False


def _tokens_by_id(tokens: list[Token]) -> dict[int, Token]:
    return {t.id: t for t in tokens}


def _value_in_token_text(value: str, cited: list[Token]) -> bool:
    if not cited:
        return False
    joined = collapse_whitespace(" ".join(t.text for t in cited))
    normalized_value = collapse_whitespace(value)
    return normalized_value in joined or joined in normalized_value


def align_field(
    raw: RawFieldExtraction,
    tokens: list[Token],
    *,
    page_width: int,
    page_height: int,
) -> AlignedField:
    token_map = _tokens_by_id(tokens)
    cited = [token_map[tid] for tid in raw.token_ids if tid in token_map]

    field = FieldValue()
    mean_ocr: float | None = None
    token_grounded = True

    if raw.value is not None:
        field.value = collapse_whitespace(raw.value)

    spans_pages = False
    if cited:
        pages = {t.page for t in cited}
        spans_pages = len(pages) > 1
        bbox_tokens = cited
        if spans_pages:
            first_page = min(pages)
            bbox_tokens = [t for t in cited if t.page == first_page]
        union = union_bbox(bbox_tokens)
        if union:
            x1, y1, x2, y2, page = union
            field.bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2, page=page)
        mean_ocr = sum(t.confidence for t in cited) / len(cited)
        value_in_tokens = _value_in_token_text(field.value or "", cited)
    elif raw.normalized_bbox is not None and raw.page is not None:
        nx, ny, nw, nh = raw.normalized_bbox
        px1 = int(nx * page_width)
        py1 = int(ny * page_height)
        px2 = px1 + max(1, int(nw * page_width))
        py2 = py1 + max(1, int(nh * page_height))
        field.bbox = BBox(x1=px1, y1=py1, x2=px2, y2=py2, page=raw.page)
        value_in_tokens = False
        token_grounded = False
        mean_ocr = None
    else:
        value_in_tokens = field.value is None
        if field.value is not None:
            token_grounded = False

    return AlignedField(
        field=field,
        mean_ocr_confidence=mean_ocr,
        model_confidence=raw.confidence,
        value_in_tokens=value_in_tokens,
        token_grounded=token_grounded,
        spans_pages=spans_pages,
    )
