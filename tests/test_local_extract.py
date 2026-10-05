from __future__ import annotations

from invoice_extract.local_extract import RulesExtractor
from invoice_extract.render import DocumentRender, PageRender, Token
from PIL import Image


def _doc_with_tokens(tokens: list[Token]) -> DocumentRender:
    img = Image.new("RGB", (400, 400), "white")
    page = PageRender(page=1, width=400, height=400, image=img, tokens=tokens)
    return DocumentRender(document_id="test.pdf", source_path=__file__, pages=[page])


def test_rules_find_invoice_number():
    tokens = [
        Token(id=0, text="Invoice", page=1, x=10, y=10, width=50, height=12, confidence=1.0),
        Token(id=1, text="No:", page=1, x=70, y=10, width=20, height=12, confidence=1.0),
        Token(id=2, text="ABC-99", page=1, x=100, y=10, width=60, height=12, confidence=1.0),
    ]
    header, _ = RulesExtractor().extract(_doc_with_tokens(tokens))
    assert header["invoiceNumber"].value == "ABC-99"
    assert header["invoiceNumber"].token_ids == [2]
