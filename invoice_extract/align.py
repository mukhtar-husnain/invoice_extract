"""Align extracted values to cited token boxes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from invoice_extract.normalize import normalize_text
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
    joined = normalize_text(" ".join(t.text for t in cited))
    normalized_value = normalize_text(value)
    if not normalized_value or not joined:
        return False
    return normalized_value in joined or (len(joined) >= 0.6 * len(normalized_value) and joined in normalized_value)


def _reading_orders(tokens: list[Token]) -> list[list[Token]]:
    by_line = sorted(tokens, key=lambda t: (t.page, round(t.y / 10), t.x))
    return [tokens, by_line]


def find_value_span(value: str, tokens: list[Token], min_ratio: float = 0.9) -> list[Token]:
    """Find the consecutive page words whose text best matches value, for re-grounding uncited values."""
    target = normalize_text(value)
    if not target:
        return []
    width = max(1, len(target.split()))
    best: tuple[float, list[Token]] = (0.0, [])
    for order in _reading_orders(tokens):
        for start in range(len(order)):
            for size in range(max(1, width - 1), width + 3):
                window = order[start : start + size]
                if len(window) < size or len({t.page for t in window}) > 1:
                    break
                joined = normalize_text(" ".join(t.text for t in window))
                if not joined:
                    continue
                if joined == target:
                    return window
                if abs(len(joined) - len(target)) > max(4, len(target) // 3):
                    continue
                matcher = SequenceMatcher(None, joined, target)
                if matcher.real_quick_ratio() < min_ratio or matcher.quick_ratio() < min_ratio:
                    continue
                ratio = matcher.ratio()
                if ratio > best[0]:
                    best = (ratio, window)
    return best[1] if best[0] >= min_ratio else []


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

    if raw.value is not None and raw.value.strip():
        field.value = collapse_whitespace(raw.value)

    if field.value is not None and not _value_in_token_text(field.value, cited):
        regrounded = find_value_span(field.value, tokens)
        if regrounded:
            cited = regrounded

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
