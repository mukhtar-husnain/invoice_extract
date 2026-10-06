"""Review flags, format validation, and arithmetic consistency checks."""

from __future__ import annotations

from typing import Iterable

from invoice_extract.config import (
    AMOUNT_FIELDS,
    DATE_FIELDS,
    MANDATORY_HEADER_FIELDS,
    REVIEW_REASON_AMOUNT_INCONSISTENT,
    REVIEW_REASON_FIELD_MISSING,
    REVIEW_REASON_INVALID_FORMAT,
    REVIEW_REASON_LOW_CONFIDENCE,
    ReviewThresholds,
)
from invoice_extract.normalize import (
    looks_like_amount,
    looks_like_text,
    normalize_currency,
    parse_amount,
    parse_date,
    parse_rate,
)
from invoice_extract.schema import FieldValue, InvoiceDocument, LineItem


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


def flag_invalid_format(field: FieldValue) -> None:
    _add_reason(field, REVIEW_REASON_INVALID_FORMAT)


NAME_FIELDS = frozenset({"supplierName", "buyerName"})


def _digit_count(value: str) -> int:
    return sum(c.isdigit() for c in value)


def value_has_valid_format(field_name: str, value: str) -> bool:
    if field_name in DATE_FIELDS:
        return parse_date(value) is not None
    if field_name in AMOUNT_FIELDS:
        return looks_like_amount(value, allow_unit=field_name == "quantity")
    if field_name == "currency":
        return normalize_currency(value) is not None
    if field_name == "taxRate":
        rate = parse_rate(value)
        return rate is not None and 0 <= rate <= 100
    if field_name == "invoiceNumber":
        return _digit_count(value) >= 3
    if field_name == "buyerTaxId":
        return _digit_count(value) >= 6 and looks_like_text(value, min_letter_ratio=0.0)
    if field_name in NAME_FIELDS:
        return looks_like_text(value)
    if field_name == "description":
        return looks_like_text(value, min_letter_ratio=0.01)
    return True


SHARED_TOKEN_PAIRS_ALLOWED = frozenset({frozenset({"totalAmount", "amountDue"})})


def check_shared_boxes(doc: InvoiceDocument) -> None:
    """Two header fields read from the exact same box means at least one of them is the wrong field."""
    by_box: dict[tuple, list[str]] = {}
    for name, field in doc.iter_header_fields():
        if field.bbox is not None:
            key = (field.bbox.page, field.bbox.x1, field.bbox.y1, field.bbox.x2, field.bbox.y2)
            by_box.setdefault(key, []).append(name)
    for names in by_box.values():
        if len(names) > 1 and frozenset(names) not in SHARED_TOKEN_PAIRS_ALLOWED:
            for name in names:
                flag_low_confidence(getattr(doc.invoice, name))


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
    field_name: str | None = None,
) -> None:
    if field.value is None or field.value.strip() == "":
        field.value = None
        field.bbox = None
        if is_mandatory:
            flag_field_missing(field)
        return

    if mean_ocr_confidence is not None and mean_ocr_confidence < thresholds.ocr_confidence:
        flag_low_confidence(field)
    if model_confidence is not None and model_confidence < thresholds.model_confidence:
        flag_low_confidence(field)
    if not value_in_tokens or not token_grounded or spans_pages:
        flag_low_confidence(field)
    if field.bbox is None:
        flag_low_confidence(field)
    if field_name and not value_has_valid_format(field_name, field.value):
        flag_invalid_format(field)


def _amounts_close(a: float | None, b: float | None, tolerance: float) -> bool:
    if a is None or b is None:
        return True
    return abs(a - b) <= max(tolerance, abs(b) * tolerance)


def _close_to_any(value: float, candidates: list[float | None], tolerance: float) -> bool:
    present = [c for c in candidates if c is not None]
    return not present or any(abs(value - c) <= max(tolerance, abs(c) * tolerance) for c in present)


def check_line_item_amounts(item: LineItem, thresholds: ReviewThresholds) -> None:
    qty = parse_amount(item.quantity.value)
    unit = parse_amount(item.unitPrice.value)
    line = parse_amount(item.lineAmount.value)
    line_tax = parse_amount(item.taxAmount.value)
    rate = parse_rate(item.taxRate.value)
    tol = thresholds.amount_tolerance

    if qty is not None and unit is not None and line is not None:
        base = qty * unit
        candidates = [base]
        if line_tax is not None:
            candidates.append(base + line_tax)
        if rate is not None:
            candidates.append(base * (1 + rate / 100))
        if not _close_to_any(line, candidates, tol):
            flag_amount_inconsistent(item.lineAmount)
            flag_amount_inconsistent(item.quantity)
            flag_amount_inconsistent(item.unitPrice)

    if rate is not None and line_tax is not None and line is not None:
        base = qty * unit if qty is not None and unit is not None else line
        expected = [base * rate / 100, line - line / (1 + rate / 100)]
        if not _close_to_any(line_tax, expected, tol):
            flag_amount_inconsistent(item.taxAmount)
            flag_amount_inconsistent(item.taxRate)


def check_document_amounts(doc: InvoiceDocument, thresholds: ReviewThresholds) -> None:
    tol = thresholds.amount_tolerance
    line_sum = 0.0
    line_count = 0
    for item in doc.lineItems:
        check_line_item_amounts(item, thresholds)
        parsed = parse_amount(item.lineAmount.value)
        if parsed is not None:
            line_sum += parsed
            line_count += 1

    tax = parse_amount(doc.invoice.taxAmount.value)
    total = parse_amount(doc.invoice.totalAmount.value)
    due = parse_amount(doc.invoice.amountDue.value)

    if line_count and line_count == len(doc.lineItems) and total is not None:
        candidates = [total, total - tax if tax is not None else None, due]
        if not _close_to_any(line_sum, candidates, tol):
            flag_amount_inconsistent(doc.invoice.totalAmount)
            for item in doc.lineItems:
                flag_amount_inconsistent(item.lineAmount)

    if tax is not None and total is not None and tax >= total:
        flag_amount_inconsistent(doc.invoice.taxAmount)

    if due is not None and total is not None and due > total * (1 + tol) + tol:
        flag_amount_inconsistent(doc.invoice.amountDue)


def rollup_document_review(doc: InvoiceDocument) -> None:
    reasons: set[str] = set()
    for name, field in doc.iter_all_fields():
        if field.isHumanReviewRequired:
            reasons.update(field.reviewReasons)
    for name in MANDATORY_HEADER_FIELDS:
        field = getattr(doc.invoice, name)
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
        is_line_field = name.startswith("lineItems")
        base_name = name.split(".")[-1] if is_line_field else name
        apply_field_review_flags(
            field,
            is_mandatory=base_name in MANDATORY_HEADER_FIELDS and not is_line_field,
            mean_ocr_confidence=meta.get("mean_ocr_confidence"),
            model_confidence=meta.get("model_confidence"),
            value_in_tokens=meta.get("value_in_tokens", True),
            token_grounded=meta.get("token_grounded", True),
            spans_pages=meta.get("spans_pages", False),
            thresholds=thresholds,
            field_name=base_name,
        )

    check_document_amounts(doc, thresholds)
    check_shared_boxes(doc)
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
