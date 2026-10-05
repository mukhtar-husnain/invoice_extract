"""Shared thresholds and environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewThresholds:
    ocr_confidence: float = 0.60
    model_confidence: float = 0.75
    amount_tolerance: float = 0.02


@dataclass(frozen=True)
class PipelineConfig:
    dpi: int = 200
    min_native_words: int = 8
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5vl:7b"
    llm_provider: str = "openai"
    image_max_side: int = 1600
    review: ReviewThresholds = ReviewThresholds()

    @classmethod
    def from_env(cls) -> PipelineConfig:
        return cls(
            ollama_base_url=os.environ.get("INVOICE_OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_model=os.environ.get("INVOICE_OLLAMA_MODEL", "qwen2.5vl:7b"),
            llm_provider=os.environ.get("INVOICE_LLM_PROVIDER", "openai").lower(),
        )


MANDATORY_HEADER_FIELDS = frozenset(
    {
        "invoiceNumber",
        "invoiceDate",
        "supplierName",
        "buyerName",
        "currency",
        "totalAmount",
    }
)

HEADER_FIELD_NAMES = (
    "invoiceNumber",
    "invoiceDate",
    "supplierName",
    "supplierAddress",
    "buyerName",
    "buyerAddress",
    "currency",
    "subtotalAmount",
    "taxAmount",
    "totalAmount",
    "paymentTerms",
)

LINE_ITEM_FIELD_NAMES = (
    "description",
    "quantity",
    "unitPrice",
    "lineAmount",
    "itemCode",
    "unitOfMeasure",
)

REVIEW_REASON_FIELD_MISSING = "FIELD_MISSING"
REVIEW_REASON_LOW_CONFIDENCE = "LOW_CONFIDENCE"
REVIEW_REASON_AMOUNT_INCONSISTENT = "AMOUNT_INCONSISTENT"
