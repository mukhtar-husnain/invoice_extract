from __future__ import annotations

from invoice_extract.config import (
    REVIEW_REASON_AMOUNT_INCONSISTENT,
    REVIEW_REASON_FIELD_MISSING,
    REVIEW_REASON_LOW_CONFIDENCE,
    ReviewThresholds,
)
from invoice_extract.review import (
    apply_document_review,
    check_line_item_amounts,
    count_fields_requiring_review,
    rollup_document_review,
)
from invoice_extract.schema import FieldValue, InvoiceDocument, LineItem


def _field(value: str | None = None, flagged: bool = False, reasons: list[str] | None = None) -> FieldValue:
    return FieldValue(
        value=value,
        isHumanReviewRequired=flagged,
        reviewReasons=reasons or [],
    )


def test_missing_mandatory_field_flags_document():
    doc = InvoiceDocument(documentId="x.pdf")
    doc.invoice.invoiceNumber = _field(None)
    doc.invoice.invoiceDate = _field("2024-01-01")
    doc.invoice.supplierName = _field("Acme")
    doc.invoice.buyerName = _field("Buyer")
    doc.invoice.currency = _field("USD")
    doc.invoice.totalAmount = _field("100.00")

    apply_document_review(doc, {"invoiceNumber": {"value_in_tokens": True, "token_grounded": True}}, ReviewThresholds())
    assert doc.invoice.invoiceNumber.isHumanReviewRequired
    assert REVIEW_REASON_FIELD_MISSING in doc.invoice.invoiceNumber.reviewReasons
    assert doc.isHumanReviewRequired
    assert REVIEW_REASON_FIELD_MISSING in doc.reviewReasons


def test_low_confidence_from_ocr_threshold():
    doc = InvoiceDocument(documentId="x.pdf")
    doc.invoice.invoiceNumber = _field("INV-1")
    meta = {
        "invoiceNumber": {
            "mean_ocr_confidence": 0.4,
            "model_confidence": 0.9,
            "value_in_tokens": True,
            "token_grounded": True,
        }
    }
    apply_document_review(doc, meta, ReviewThresholds())
    assert REVIEW_REASON_LOW_CONFIDENCE in doc.invoice.invoiceNumber.reviewReasons


def test_amount_inconsistency_flags_line_item():
    item = LineItem(
        quantity=_field("3"),
        unitPrice=_field("10.00"),
        lineAmount=_field("25.00"),
    )
    check_line_item_amounts(item, ReviewThresholds(amount_tolerance=0.01))
    assert item.lineAmount.isHumanReviewRequired
    assert REVIEW_REASON_AMOUNT_INCONSISTENT in item.lineAmount.reviewReasons


def test_cross_page_field_gets_low_confidence():
    doc = InvoiceDocument(documentId="x.pdf")
    doc.invoice.supplierAddress = _field("line one continues")
    meta = {
        "supplierAddress": {
            "mean_ocr_confidence": 0.95,
            "model_confidence": 0.95,
            "value_in_tokens": True,
            "token_grounded": True,
            "spans_pages": True,
        }
    }
    apply_document_review(doc, meta, ReviewThresholds())
    assert REVIEW_REASON_LOW_CONFIDENCE in doc.invoice.supplierAddress.reviewReasons


def test_field_count_excludes_document_only_flags():
    doc = InvoiceDocument(documentId="x.pdf")
    doc.invoice.invoiceNumber = _field("A", flagged=True, reasons=[REVIEW_REASON_LOW_CONFIDENCE])
    doc.invoice.totalAmount = _field("10", flagged=True, reasons=[REVIEW_REASON_LOW_CONFIDENCE])
    rollup_document_review(doc)
    assert count_fields_requiring_review(doc) == 2
