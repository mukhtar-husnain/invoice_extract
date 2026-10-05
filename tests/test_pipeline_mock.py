from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from invoice_extract.pipeline import run_pipeline


def test_run_pipeline_rules_writes_summary(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "output"

    fake_doc_json = {
        "documentId": "a.pdf",
        "invoice": {
            "invoiceNumber": {"value": "1", "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
            "invoiceDate": {"value": None, "bbox": None, "isHumanReviewRequired": True, "reviewReasons": ["FIELD_MISSING"]},
            "supplierName": {"value": "S", "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
            "supplierAddress": {"value": None, "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
            "buyerName": {"value": "B", "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
            "buyerAddress": {"value": None, "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
            "currency": {"value": "USD", "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
            "subtotalAmount": {"value": None, "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
            "taxAmount": {"value": None, "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
            "totalAmount": {"value": "1", "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
            "paymentTerms": {"value": None, "bbox": None, "isHumanReviewRequired": False, "reviewReasons": []},
        },
        "lineItems": [],
        "isHumanReviewRequired": True,
        "reviewReasons": ["FIELD_MISSING"],
    }

    from invoice_extract.schema import InvoiceDocument

    fake_doc = InvoiceDocument.model_validate(fake_doc_json)

    with patch("invoice_extract.pipeline.discover_pdfs", return_value=[data_dir / "a.pdf"]):
        with patch("invoice_extract.pipeline.process_pdf", return_value=fake_doc):
            summary = run_pipeline(data_dir, out_dir, extractor_mode="rules")

    assert summary.extractor == "rules"
    assert (out_dir / "a.json").exists()
    assert (out_dir / "summary.json").exists()
    assert summary.fieldsRequiringHumanReview == 1
    assert summary.invoicesRequiringHumanReview == 1

    import json

    written = json.loads((out_dir / "a.json").read_text())
    assert "invoice" in written
    assert "invoiceNumber" in written["invoice"]
