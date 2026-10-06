from __future__ import annotations

import argparse
import sys
from pathlib import Path

from invoice_extract.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract structured fields from invoice PDFs.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Directory containing PDF invoices")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory for JSON results")
    parser.add_argument(
        "--extractor",
        choices=("auto", "ollama", "api", "rules"),
        default="auto",
        help="Extraction backend: local Ollama, external vision API, label rules, or auto-detect",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data.exists():
        print(f"Data directory not found: {args.data}", file=sys.stderr)
        return 1
    try:
        summary = run_pipeline(args.data, args.out, extractor_mode=args.extractor)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"Processed {summary.documentsProcessed} documents with extractor={summary.extractor}. "
        f"Fields requiring review: {summary.fieldsRequiringHumanReview}. "
        f"Invoices requiring review: {summary.invoicesRequiringHumanReview}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
