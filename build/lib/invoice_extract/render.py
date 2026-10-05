"""Rasterize PDF pages and extract native text-layer word boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz
from PIL import Image


@dataclass
class Token:
    id: int
    text: str
    page: int
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class PageRender:
    page: int
    width: int
    height: int
    image: Image.Image
    tokens: list[Token] = field(default_factory=list)


@dataclass
class DocumentRender:
    document_id: str
    source_path: Path
    pages: list[PageRender] = field(default_factory=list)

    @property
    def tokens(self) -> list[Token]:
        out: list[Token] = []
        for page in self.pages:
            out.extend(page.tokens)
        return out


def _word_to_pixels(word: fitz.Rect, scale: float) -> tuple[int, int, int, int]:
    x0 = int(word.x0 * scale)
    y0 = int(word.y0 * scale)
    x1 = int(word.x1 * scale)
    y1 = int(word.y1 * scale)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _extract_native_words(page: fitz.Page, page_num: int, scale: float, start_id: int) -> list[Token]:
    tokens: list[Token] = []
    words = page.get_text("words")
    for idx, word in enumerate(words):
        if len(word) < 5:
            continue
        x, y, w, h = _word_to_pixels(fitz.Rect(word[0], word[1], word[2], word[3]), scale)
        text = str(word[4]).strip()
        if not text:
            continue
        tokens.append(
            Token(
                id=start_id + idx,
                text=text,
                page=page_num,
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=1.0,
            )
        )
    return tokens


def render_pdf(path: Path, *, dpi: int = 200) -> DocumentRender:
    document_id = path.name
    doc = fitz.open(path)
    scale = dpi / 72.0
    pages: list[PageRender] = []
    next_id = 0

    for page_index in range(len(doc)):
        page_num = page_index + 1
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        native = _extract_native_words(page, page_num, scale, next_id)
        next_id += len(native)
        pages.append(
            PageRender(
                page=page_num,
                width=pix.width,
                height=pix.height,
                image=image,
                tokens=native,
            )
        )

    doc.close()
    return DocumentRender(document_id=document_id, source_path=path, pages=pages)


def union_bbox(tokens: list[Token]) -> tuple[int, int, int, int, int] | None:
    if not tokens:
        return None
    x1 = min(t.x for t in tokens)
    y1 = min(t.y for t in tokens)
    x2 = max(t.x + t.width for t in tokens)
    y2 = max(t.y + t.height for t in tokens)
    page = tokens[0].page
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2), page
