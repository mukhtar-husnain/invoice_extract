from __future__ import annotations

from invoice_extract.align import RawFieldExtraction, align_field, collapse_whitespace
from invoice_extract.render import Token, union_bbox


def test_collapse_whitespace():
    assert collapse_whitespace("  hello   world ") == "hello world"


def test_union_bbox(sample_tokens):
    cited = sample_tokens[0:3]
    x1, y1, x2, y2, page = union_bbox(cited)
    assert page == 1
    assert x1 == 10
    assert y1 == 10
    assert x2 == 160
    assert y2 == 22


def test_align_field_token_grounded(sample_tokens):
    raw = RawFieldExtraction(value="INV-001", token_ids=[2], confidence=0.95)
    aligned = align_field(raw, sample_tokens, page_width=800, page_height=1100)
    assert aligned.field.value == "INV-001"
    assert aligned.field.bbox is not None
    assert aligned.field.bbox.x1 == 100
    assert aligned.field.bbox.x2 == 160
    assert aligned.token_grounded is True
    assert aligned.value_in_tokens is True


def test_align_field_cross_page_uses_first_page_bbox():
    tokens = [
        Token(id=0, text="Long", page=1, x=10, y=10, width=40, height=12, confidence=1.0),
        Token(id=1, text="value", page=2, x=10, y=10, width=40, height=12, confidence=1.0),
    ]
    raw = RawFieldExtraction(value="Long value", token_ids=[0, 1], confidence=0.9)
    aligned = align_field(raw, tokens, page_width=800, page_height=1100)
    assert aligned.spans_pages is True
    assert aligned.field.bbox is not None
    assert aligned.field.bbox.page == 1
    assert aligned.field.bbox.x1 == 10
    assert aligned.field.bbox.x2 == 50


def test_align_field_ungrounded_bbox(sample_tokens):
    raw = RawFieldExtraction(
        value="guess",
        token_ids=[],
        confidence=0.5,
        normalized_bbox=(0.1, 0.2, 0.3, 0.05),
        page=1,
    )
    aligned = align_field(raw, sample_tokens, page_width=1000, page_height=1000)
    assert aligned.field.bbox is not None
    assert aligned.field.bbox.x1 == 100
    assert aligned.field.bbox.y1 == 200
    assert aligned.field.bbox.x2 == 400
    assert aligned.field.bbox.y2 == 250
    assert aligned.token_grounded is False
    assert aligned.value_in_tokens is False
