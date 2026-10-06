from __future__ import annotations

from invoice_extract.config import (
    REVIEW_REASON_AMOUNT_INCONSISTENT,
    REVIEW_REASON_FIELD_MISSING,
    REVIEW_REASON_LOW_CONFIDENCE,
    ReviewThresholds,
)
from invoice_extract.review import (
    apply_document_review,
    check_document_amounts,
    check_line_item_amounts,
    check_shared_boxes,
    count_fields_requiring_review,
    rollup_document_review,
    value_has_valid_format,
)
from invoice_extract.schema import BBox, FieldValue, InvoiceDocument, LineItem


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


def test_tax_id_read_as_tax_amount_is_invalid_format():
    assert not value_has_valid_format("taxAmount", "615-734-218-00000")
    assert not value_has_valid_format("totalAmount", "9%")
    assert not value_has_valid_format("invoiceNumber", "82")
    assert not value_has_valid_format("buyerTaxId", "\\2")
    assert value_has_valid_format("buyerTaxId", "008-183-175-000")
    assert value_has_valid_format("quantity", "1 PK100")


def test_two_fields_on_the_same_box_are_both_flagged():
    box = BBox(page=1, x1=10, y1=10, x2=50, y2=20)
    doc = InvoiceDocument(documentId="x.pdf")
    doc.invoice.taxAmount = FieldValue(value="615-734-218", bbox=box)
    doc.invoice.buyerTaxId = FieldValue(value="615-734-218", bbox=box)
    doc.invoice.totalAmount = FieldValue(value="100.00", bbox=BBox(page=1, x1=60, y1=10, x2=90, y2=20))
    doc.invoice.amountDue = FieldValue(value="100.00", bbox=BBox(page=1, x1=60, y1=10, x2=90, y2=20))
    check_shared_boxes(doc)
    assert REVIEW_REASON_LOW_CONFIDENCE in doc.invoice.taxAmount.reviewReasons
    assert REVIEW_REASON_LOW_CONFIDENCE in doc.invoice.buyerTaxId.reviewReasons
    assert not doc.invoice.totalAmount.isHumanReviewRequired


def test_lines_not_adding_up_flag_every_line_amount():
    doc = InvoiceDocument(documentId="x.pdf")
    doc.invoice.totalAmount = _field("221.82")
    doc.lineItems = [LineItem(lineAmount=_field("20")), LineItem(lineAmount=_field("4"))]
    check_document_amounts(doc, ReviewThresholds())
    assert all(REVIEW_REASON_AMOUNT_INCONSISTENT in i.lineAmount.reviewReasons for i in doc.lineItems)


def test_field_count_excludes_document_only_flags():
    doc = InvoiceDocument(documentId="x.pdf")
    doc.invoice.invoiceNumber = _field("A", flagged=True, reasons=[REVIEW_REASON_LOW_CONFIDENCE])
    doc.invoice.totalAmount = _field("10", flagged=True, reasons=[REVIEW_REASON_LOW_CONFIDENCE])
    rollup_document_review(doc)
    assert count_fields_requiring_review(doc) == 2
