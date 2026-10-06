from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    page: int = Field(ge=1)

    @model_validator(mode="after")
    def corners_are_ordered(self) -> BBox:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bbox corners must satisfy x1 <= x2 and y1 <= y2")
        return self


class FieldValue(BaseModel):
    value: str | None = None
    bbox: BBox | None = None
    isHumanReviewRequired: bool = False
    reviewReasons: list[str] = Field(default_factory=list)


class LineItem(BaseModel):
    description: FieldValue = Field(default_factory=FieldValue)
    quantity: FieldValue = Field(default_factory=FieldValue)
    unitPrice: FieldValue = Field(default_factory=FieldValue)
    taxRate: FieldValue = Field(default_factory=FieldValue)
    taxAmount: FieldValue = Field(default_factory=FieldValue)
    lineAmount: FieldValue = Field(default_factory=FieldValue)

    def iter_fields(self) -> list[tuple[str, FieldValue]]:
        from invoice_extract.config import LINE_ITEM_FIELD_NAMES

        return [(name, getattr(self, name)) for name in LINE_ITEM_FIELD_NAMES]


class InvoiceHeader(BaseModel):
    invoiceNumber: FieldValue = Field(default_factory=FieldValue)
    invoiceDate: FieldValue = Field(default_factory=FieldValue)
    dueDate: FieldValue = Field(default_factory=FieldValue)
    supplierName: FieldValue = Field(default_factory=FieldValue)
    supplierAddress: FieldValue = Field(default_factory=FieldValue)
    buyerName: FieldValue = Field(default_factory=FieldValue)
    buyerTaxId: FieldValue = Field(default_factory=FieldValue)
    currency: FieldValue = Field(default_factory=FieldValue)
    taxAmount: FieldValue = Field(default_factory=FieldValue)
    totalAmount: FieldValue = Field(default_factory=FieldValue)
    amountDue: FieldValue = Field(default_factory=FieldValue)
    paymentTerms: FieldValue = Field(default_factory=FieldValue)
    purchaseOrderNumber: FieldValue = Field(default_factory=FieldValue)


class InvoiceDocument(BaseModel):
    documentId: str
    invoice: InvoiceHeader = Field(default_factory=InvoiceHeader)
    lineItems: list[LineItem] = Field(default_factory=list)
    isHumanReviewRequired: bool = False
    reviewReasons: list[str] = Field(default_factory=list)

    def iter_header_fields(self) -> list[tuple[str, FieldValue]]:
        from invoice_extract.config import HEADER_FIELD_NAMES

        return [(name, getattr(self.invoice, name)) for name in HEADER_FIELD_NAMES]

    def iter_all_fields(self) -> list[tuple[str, FieldValue]]:
        fields = list(self.iter_header_fields())
        for idx, item in enumerate(self.lineItems):
            for name, fv in item.iter_fields():
                fields.append((f"lineItems[{idx}].{name}", fv))
        return fields

    def to_output_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ExtractionSummary(BaseModel):
    extractor: str
    documentsProcessed: int
    fieldsRequiringHumanReview: int
    invoicesRequiringHumanReview: int
    documents: list[str] = Field(default_factory=list)
