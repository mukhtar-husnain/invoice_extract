from __future__ import annotations

import json

from invoice_extract.schema import BBox, FieldValue, InvoiceDocument, InvoiceHeader, LineItem


def test_schema_round_trip():
    doc = InvoiceDocument(
        documentId="digital_01.pdf",
        invoice=InvoiceHeader(
            invoiceNumber=FieldValue(value="INV-1", bbox=BBox(x1=1, y1=2, x2=4, y2=6, page=1)),
        ),
        lineItems=[
            LineItem(description=FieldValue(value="Widget")),
        ],
    )
    payload = doc.to_output_dict()
    restored = InvoiceDocument.model_validate(payload)
    assert restored.documentId == "digital_01.pdf"
    assert restored.invoice.invoiceNumber.bbox is not None
    assert restored.invoice.invoiceNumber.bbox.x2 - restored.invoice.invoiceNumber.bbox.x1 == 3
    assert restored.invoice.invoiceNumber.bbox.y2 - restored.invoice.invoiceNumber.bbox.y1 == 4
    assert "invoice" in payload
    assert "invoiceNumber" not in payload
    assert json.dumps(payload)


def test_output_shape_matches_contract():
    doc = InvoiceDocument(documentId="invoice_001.pdf")
    payload = doc.to_output_dict()
    assert set(payload.keys()) == {
        "documentId",
        "invoice",
        "lineItems",
        "isHumanReviewRequired",
        "reviewReasons",
    }
    assert set(payload["invoice"].keys()) == {
        "invoiceNumber",
        "invoiceDate",
        "dueDate",
        "supplierName",
        "supplierAddress",
        "buyerName",
        "buyerTaxId",
        "currency",
        "taxAmount",
        "totalAmount",
        "amountDue",
        "paymentTerms",
        "purchaseOrderNumber",
    }
    for field in payload["invoice"].values():
        assert set(field.keys()) == {"value", "bbox", "isHumanReviewRequired", "reviewReasons"}


def test_line_item_shape_matches_contract():
    payload = LineItem().model_dump(mode="json")
    assert set(payload.keys()) == {"description", "quantity", "unitPrice", "taxRate", "taxAmount", "lineAmount"}
