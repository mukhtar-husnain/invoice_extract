"""Review flags and arithmetic consistency checks."""

from __future__ import annotations

import re
from typing import Iterable

from invoice_extract.config import (
    MANDATORY_HEADER_FIELDS,
    REVIEW_REASON_AMOUNT_INCONSISTENT,
    REVIEW_REASON_FIELD_MISSING,
    REVIEW_REASON_LOW_CONFIDENCE,
    ReviewThresholds,
)
from invoice_extract.schema import FieldValue, InvoiceDocument, LineItem


def _parse_amount(text: str | None) -> float | None:
    if not text:
        return None
    cleaned = re.sub(r"[^\d.,\-]", "", text.replace(",", ""))
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _add_reason(field: FieldValue, reason: str) -> None:
    if reason not in field.reviewReasons:
        field.reviewReasons.append(reason)
    field.isHumanReviewRequired = True


def flag_field_missing(field: FieldValue) -> None:
    _add_reason(field, REVIEW_REASON_FIELD_MISSING)


def flag_low_confidence(field: FieldValue) -> None:
    _add_reason(field, REVIEW_REASON_LOW_CONFIDENCE)


def flag_amount_inconsistent(field: FieldValue) -> None:
    _add_reason(field, REVIEW_REASON_AMOUNT_INCONSISTENT)


def apply_field_review_flags(
    field: FieldValue,
    *,
    is_mandatory: bool,
    mean_ocr_confidence: float | None,
    model_confidence: float | None,
    value_in_tokens: bool,
    token_grounded: bool,
    spans_pages: bool = False,
    thresholds: ReviewThresholds,
) -> None:
    if is_mandatory and (field.value is None or field.value.strip() == ""):
        flag_field_missing(field)
        return

    if field.value is None:
        return

    if mean_ocr_confidence is not None and mean_ocr_confidence < thresholds.ocr_confidence:
        flag_low_confidence(field)
    if model_confidence is not None and model_confidence < thresholds.model_confidence:
        flag_low_confidence(field)
    if not value_in_tokens:
        flag_low_confidence(field)
    if not token_grounded:
        flag_low_confidence(field)
    if spans_pages:
        flag_low_confidence(field)


def _amounts_close(a: float | None, b: float | None, tolerance: float) -> bool:
    if a is None or b is None:
        return True
    return abs(a - b) <= max(tolerance, abs(a) * tolerance)


def check_line_item_amounts(item: LineItem, thresholds: ReviewThresholds) -> None:
    qty = _parse_amount(item.quantity.value)
    unit = _parse_amount(item.unitPrice.value)
    line = _parse_amount(item.lineAmount.value)
    if qty is not None and unit is not None and line is not None:
        expected = qty * unit
        if not _amounts_close(expected, line, thresholds.amount_tolerance):
            flag_amount_inconsistent(item.lineAmount)
            flag_amount_inconsistent(item.quantity)
            flag_amount_inconsistent(item.unitPrice)


def check_document_amounts(doc: InvoiceDocument, thresholds: ReviewThresholds) -> None:
    line_sum = 0.0
    line_count = 0
    for item in doc.lineItems:
        check_line_item_amounts(item, thresholds)
        parsed = _parse_amount(item.lineAmount.value)
        if parsed is not None:
            line_sum += parsed
            line_count += 1

    subtotal = _parse_amount(doc.invoice.subtotalAmount.value)
    tax = _parse_amount(doc.invoice.taxAmount.value)
    total = _parse_amount(doc.invoice.totalAmount.value)

    if line_count > 0 and subtotal is not None:
        if not _amounts_close(line_sum, subtotal, thresholds.amount_tolerance):
            flag_amount_inconsistent(doc.invoice.subtotalAmount)

    if subtotal is not None and tax is not None and total is not None:
        if not _amounts_close(subtotal + tax, total, thresholds.amount_tolerance):
            flag_amount_inconsistent(doc.invoice.totalAmount)
            flag_amount_inconsistent(doc.invoice.taxAmount)


def rollup_document_review(doc: InvoiceDocument) -> None:
    reasons: set[str] = set()
    for name, field in doc.iter_all_fields():
        if field.isHumanReviewRequired:
            reasons.update(field.reviewReasons)
        if name.split(".")[0] in MANDATORY_HEADER_FIELDS or name in MANDATORY_HEADER_FIELDS:
            if field.value is None or field.value.strip() == "":
                reasons.add(REVIEW_REASON_FIELD_MISSING)

    doc.reviewReasons = sorted(reasons)
    doc.isHumanReviewRequired = bool(reasons)


def apply_document_review(
    doc: InvoiceDocument,
    field_meta: dict[str, dict],
    thresholds: ReviewThresholds,
) -> None:
    for name, field in doc.iter_all_fields():
        meta = field_meta.get(name, {})
        base_name = name.split(".")[-1] if name.startswith("lineItems") else name
        is_mandatory = base_name in MANDATORY_HEADER_FIELDS and not name.startswith("lineItems")
        apply_field_review_flags(
            field,
            is_mandatory=is_mandatory,
            mean_ocr_confidence=meta.get("mean_ocr_confidence"),
            model_confidence=meta.get("model_confidence"),
            value_in_tokens=meta.get("value_in_tokens", True),
            token_grounded=meta.get("token_grounded", True),
            spans_pages=meta.get("spans_pages", False),
            thresholds=thresholds,
        )

    check_document_amounts(doc, thresholds)
    rollup_document_review(doc)


def count_fields_requiring_review(doc: InvoiceDocument) -> int:
    return sum(1 for _, field in doc.iter_all_fields() if field.isHumanReviewRequired)


def collect_review_reasons(docs: Iterable[InvoiceDocument]) -> tuple[int, int]:
    field_count = 0
    invoice_count = 0
    for doc in docs:
        field_count += count_fields_requiring_review(doc)
        if doc.isHumanReviewRequired:
            invoice_count += 1
    return field_count, invoice_count
