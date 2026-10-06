from __future__ import annotations

import sys

import pytesseract
from pytesseract import Output, TesseractNotFoundError

from invoice_extract.render import PageRender, Token

_tesseract_available: bool | None = None
_warned_missing_tesseract = False


def tesseract_is_available() -> bool:
    global _tesseract_available
    if _tesseract_available is None:
        try:
            pytesseract.get_tesseract_version()
            _tesseract_available = True
        except TesseractNotFoundError:
            _tesseract_available = False
    return _tesseract_available


def _next_token_id(pages: list[PageRender]) -> int:
    max_id = -1
    for page in pages:
        for token in page.tokens:
            max_id = max(max_id, token.id)
    return max_id + 1


def ocr_page_if_needed(page: PageRender, *, min_native_words: int, start_id: int) -> list[Token]:
    if len(page.tokens) >= min_native_words:
        return page.tokens

    if not tesseract_is_available():
        global _warned_missing_tesseract
        if not _warned_missing_tesseract:
            print(
                "Warning: Tesseract not found; image-only pages will have no OCR tokens. "
                "Install with: brew install tesseract tesseract-lang",
                file=sys.stderr,
            )
            _warned_missing_tesseract = True
        return []

    data = pytesseract.image_to_data(
        page.image,
        lang="eng+chi_sim+chi_tra+vie",
        output_type=Output.DICT,
    )
    tokens: list[Token] = []
    token_id = start_id
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf_raw = data["conf"][i]
        try:
            confidence = max(0.0, min(1.0, float(conf_raw) / 100.0))
        except (TypeError, ValueError):
            confidence = 0.5
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        tokens.append(
            Token(
                id=token_id,
                text=text,
                page=page.page,
                x=x,
                y=y,
                width=max(1, w),
                height=max(1, h),
                confidence=confidence,
            )
        )
        token_id += 1
    return tokens


def apply_ocr_fallback(pages: list[PageRender], *, min_native_words: int) -> None:
    next_id = _next_token_id(pages)
    for page in pages:
        if len(page.tokens) >= min_native_words:
            continue
        page.tokens = ocr_page_if_needed(page, min_native_words=min_native_words, start_id=next_id)
        if page.tokens:
            next_id = max(t.id for t in page.tokens) + 1
