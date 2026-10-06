from __future__ import annotations

import json
from pathlib import Path

from invoice_extract.align import align_field
from invoice_extract.config import LINE_ITEM_FIELD_NAMES, PipelineConfig
from invoice_extract.extract import (
    BaseExtractor,
    OllamaExtractor,
    api_key_is_available,
    build_api_extractor,
    default_page_size,
    ollama_is_available,
)
from invoice_extract.local_extract import RulesExtractor as LocalRulesExtractor
from invoice_extract.ocr import apply_ocr_fallback
from invoice_extract.render import render_pdf
from invoice_extract.review import apply_document_review, collect_review_reasons
from invoice_extract.schema import ExtractionSummary, FieldValue, InvoiceDocument, LineItem


def discover_pdfs(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.pdf"))


def resolve_extractor(mode: str, config: PipelineConfig) -> tuple[BaseExtractor, str]:
    if mode == "ollama":
        if not ollama_is_available(config):
            raise RuntimeError(f"Ollama is not reachable at {config.ollama_base_url}")
        return OllamaExtractor(config), "ollama"
    if mode == "api":
        if not api_key_is_available(config):
            raise RuntimeError("No API key found for the configured LLM provider")
        return build_api_extractor(config), "api"
    if mode == "rules":
        return LocalRulesExtractor(), "rules"

    if ollama_is_available(config):
        return OllamaExtractor(config), "ollama"
    if api_key_is_available(config):
        return build_api_extractor(config), "api"
    return LocalRulesExtractor(), "rules"


def _build_document(
    document_id: str,
    header_raw: dict,
    line_items_raw: list[dict],
    tokens: list,
    pages: list,
    config: PipelineConfig,
) -> tuple[InvoiceDocument, dict[str, dict]]:
    page_w, page_h = default_page_size(pages)
    doc = InvoiceDocument(documentId=document_id)
    field_meta: dict[str, dict] = {}

    for name, raw in header_raw.items():
        aligned = align_field(raw, tokens, page_width=page_w, page_height=page_h)
        setattr(doc.invoice, name, aligned.field)
        field_meta[name] = {
            "mean_ocr_confidence": aligned.mean_ocr_confidence,
            "model_confidence": aligned.model_confidence,
            "value_in_tokens": aligned.value_in_tokens,
            "token_grounded": aligned.token_grounded,
            "spans_pages": aligned.spans_pages,
        }

    doc.lineItems = []
    for idx, row in enumerate(line_items_raw):
        item = LineItem()
        for name in LINE_ITEM_FIELD_NAMES:
            aligned = align_field(row[name], tokens, page_width=page_w, page_height=page_h)
            setattr(item, name, aligned.field)
            field_meta[f"lineItems[{idx}].{name}"] = {
                "mean_ocr_confidence": aligned.mean_ocr_confidence,
                "model_confidence": aligned.model_confidence,
                "value_in_tokens": aligned.value_in_tokens,
                "token_grounded": aligned.token_grounded,
                "spans_pages": aligned.spans_pages,
            }
        doc.lineItems.append(item)

    apply_document_review(doc, field_meta, config.review)
    return doc, field_meta


def process_pdf(path: Path, extractor: BaseExtractor, config: PipelineConfig) -> InvoiceDocument:
    rendered = render_pdf(path, dpi=config.dpi)
    apply_ocr_fallback(rendered.pages, min_native_words=config.min_native_words)
    tokens = rendered.tokens
    header_raw, line_items_raw = extractor.extract(rendered)
    doc, _ = _build_document(
        rendered.document_id,
        header_raw,
        line_items_raw,
        tokens,
        rendered.pages,
        config,
    )
    return doc


def run_pipeline(data_dir: Path, out_dir: Path, extractor_mode: str = "auto") -> ExtractionSummary:
    config = PipelineConfig.from_env()
    extractor, extractor_name = resolve_extractor(extractor_mode, config)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs(data_dir)
    documents: list[InvoiceDocument] = []

    for pdf in pdfs:
        doc = process_pdf(pdf, extractor, config)
        documents.append(doc)
        out_path = out_dir / f"{doc.documentId.replace('.pdf', '')}.json"
        out_path.write_text(json.dumps(doc.to_output_dict(), indent=2), encoding="utf-8")

    fields_review, invoices_review = collect_review_reasons(documents)
    summary = ExtractionSummary(
        extractor=extractor_name,
        documentsProcessed=len(documents),
        fieldsRequiringHumanReview=fields_review,
        invoicesRequiringHumanReview=invoices_review,
        documents=[d.documentId for d in documents],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")
    return summary
