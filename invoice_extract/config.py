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
    ollama_num_ctx: int = 16384
    ollama_timeout_s: float = 1800.0
    llm_provider: str = "openai"
    image_max_side: int = 1280
    review: ReviewThresholds = ReviewThresholds()

    @classmethod
    def from_env(cls) -> PipelineConfig:
        return cls(
            ollama_base_url=env_or("INVOICE_OLLAMA_URL", cls.ollama_base_url),
            ollama_model=env_or("INVOICE_OLLAMA_MODEL", cls.ollama_model),
            ollama_num_ctx=int(env_or("INVOICE_OLLAMA_NUM_CTX", cls.ollama_num_ctx)),
            ollama_timeout_s=float(env_or("INVOICE_OLLAMA_TIMEOUT_S", cls.ollama_timeout_s)),
            llm_provider=env_or("INVOICE_LLM_PROVIDER", cls.llm_provider).lower(),
            image_max_side=int(env_or("INVOICE_IMAGE_MAX_SIDE", cls.image_max_side)),
        )


def env_or(name: str, default):
    value = os.environ.get(name, "").strip()
    return value if value else default


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
)

LINE_ITEM_FIELD_NAMES = (
    "description",
    "quantity",
    "unitPrice",
    "taxRate",
    "taxAmount",
    "lineAmount",
)

AMOUNT_FIELDS = frozenset({"taxAmount", "totalAmount", "amountDue", "quantity", "unitPrice", "lineAmount"})
DATE_FIELDS = frozenset({"invoiceDate", "dueDate"})

REVIEW_REASON_FIELD_MISSING = "FIELD_MISSING"
REVIEW_REASON_LOW_CONFIDENCE = "LOW_CONFIDENCE"
REVIEW_REASON_AMOUNT_INCONSISTENT = "AMOUNT_INCONSISTENT"
REVIEW_REASON_INVALID_FORMAT = "INVALID_FORMAT"
