"""Score pipeline output against a hand-checked answer key and compute the missed-review rate.

Missed-review rate = extracted values that are wrong and not flagged for review / all extracted values.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from invoice_extract.config import AMOUNT_FIELDS, DATE_FIELDS, HEADER_FIELD_NAMES, LINE_ITEM_FIELD_NAMES
from invoice_extract.normalize import (
    normalize_currency,
    normalize_identifier,
    normalize_text,
    parse_amount,
    parse_date,
    parse_rate,
)

IDENTIFIER_FIELDS = frozenset({"invoiceNumber", "purchaseOrderNumber", "buyerTaxId"})
TEXT_MATCH_RATIO = 0.85


def _options(gold: Any) -> list[str | None]:
    return list(gold) if isinstance(gold, list) else [gold]


def values_match(field_name: str, predicted: str, expected: str) -> bool:
    if field_name in AMOUNT_FIELDS:
        a, b = parse_amount(predicted), parse_amount(expected)
        return a is not None and b is not None and abs(a - b) < 0.005
    if field_name in DATE_FIELDS:
        a, b = parse_date(predicted), parse_date(expected)
        return a is not None and a == b
    if field_name == "currency":
        a, b = normalize_currency(predicted), normalize_currency(expected)
        return a is not None and a == b
    if field_name == "taxRate":
        a, b = parse_rate(predicted), parse_rate(expected)
        return a is not None and b is not None and abs(a - b) < 0.005
    if field_name in IDENTIFIER_FIELDS:
        return normalize_identifier(predicted) == normalize_identifier(expected) != ""
    a, b = normalize_text(predicted), normalize_text(expected)
    return bool(a and b) and (a == b or SequenceMatcher(None, a, b).ratio() >= TEXT_MATCH_RATIO)


@dataclass
class Tally:
    extracted: int = 0
    correct: int = 0
    incorrect: int = 0
    incorrect_flagged: int = 0
    incorrect_not_flagged: int = 0
    correct_flagged: int = 0
    missing: int = 0
    missing_flagged: int = 0
    expected_values: int = 0
    correct_with_bbox: int = 0

    def add(self, other: Tally) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, Any]:
        def pct(n: int, d: int) -> float | None:
            return round(100.0 * n / d, 2) if d else None

        return {
            "extractedValues": self.extracted,
            "correctValues": self.correct,
            "incorrectValues": self.incorrect,
            "incorrectFlagged": self.incorrect_flagged,
            "incorrectNotFlagged": self.incorrect_not_flagged,
            "correctButFlagged": self.correct_flagged,
            "expectedValues": self.expected_values,
            "missingValues": self.missing,
            "missingFlagged": self.missing_flagged,
            "missedReviewRatePct": pct(self.incorrect_not_flagged, self.extracted),
            "precisionPct": pct(self.correct, self.extracted),
            "recallPct": pct(self.correct, self.expected_values),
            "correctValuesWithBboxPct": pct(self.correct_with_bbox, self.correct),
        }


@dataclass
class FieldOutcome:
    document: str
    field: str
    predicted: str | None
    expected: Any
    status: str
    flagged: bool


@dataclass
class Evaluation:
    total: Tally = field(default_factory=Tally)
    by_field: dict[str, Tally] = field(default_factory=lambda: defaultdict(Tally))
    by_document: dict[str, Tally] = field(default_factory=lambda: defaultdict(Tally))
    errors: list[FieldOutcome] = field(default_factory=list)


def score_field(field_name: str, predicted: dict[str, Any] | None, gold: Any) -> tuple[Tally, str]:
    tally = Tally()
    value = (predicted or {}).get("value")
    flagged = bool((predicted or {}).get("isHumanReviewRequired"))
    options = _options(gold)
    non_null = [o for o in options if o is not None]
    if non_null and None not in options:
        tally.expected_values = 1

    if value is None or str(value).strip() == "":
        if None in options:
            return tally, "correct_empty"
        tally.missing = 1
        tally.missing_flagged = int(flagged)
        return tally, "missing"

    tally.extracted = 1
    if any(values_match(field_name, str(value), o) for o in non_null):
        tally.correct = 1
        tally.correct_flagged = int(flagged)
        tally.correct_with_bbox = int((predicted or {}).get("bbox") is not None)
        if None in options:
            tally.expected_values = 1
        return tally, "correct"
    tally.incorrect = 1
    if flagged:
        tally.incorrect_flagged = 1
        return tally, "incorrect_flagged"
    tally.incorrect_not_flagged = 1
    return tally, "incorrect_not_flagged"


def _row_similarity(pred_row: dict[str, Any], gold_row: dict[str, Any]) -> float:
    score = 0.0
    pred_desc = (pred_row.get("description") or {}).get("value")
    gold_desc = gold_row.get("description")
    if pred_desc and gold_desc:
        score += SequenceMatcher(None, normalize_text(pred_desc), normalize_text(str(gold_desc))).ratio()
    pred_amount = (pred_row.get("lineAmount") or {}).get("value")
    gold_amount = gold_row.get("lineAmount")
    if pred_amount and gold_amount and any(values_match("lineAmount", pred_amount, o) for o in _options(gold_amount) if o):
        score += 1.0
    return score


def match_line_items(pred_rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> list[tuple[int | None, int | None]]:
    """Greedy one-to-one matching of predicted rows to answer-key rows by description and amount."""
    candidates = sorted(
        ((_row_similarity(p, g), pi, gi) for pi, p in enumerate(pred_rows) for gi, g in enumerate(gold_rows)),
        reverse=True,
    )
    used_p: set[int] = set()
    used_g: set[int] = set()
    pairs: list[tuple[int | None, int | None]] = []
    for score, pi, gi in candidates:
        if score < 0.6 or pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi))
    pairs += [(pi, None) for pi in range(len(pred_rows)) if pi not in used_p]
    pairs += [(None, gi) for gi in range(len(gold_rows)) if gi not in used_g]
    return pairs


def evaluate(gold: dict[str, Any], pred_dir: Path) -> Evaluation:
    result = Evaluation()
    for document_id, expected in gold.items():
        if document_id.startswith("_"):
            continue
        pred_path = pred_dir / f"{Path(document_id).stem}.json"
        predicted = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else {}

        def record(field_name: str, label: str, pred_field: dict[str, Any] | None, gold_value: Any) -> None:
            tally, status = score_field(field_name.split(".")[-1], pred_field, gold_value)
            result.total.add(tally)
            result.by_field[field_name].add(tally)
            result.by_document[document_id].add(tally)
            if status in {"incorrect_flagged", "incorrect_not_flagged", "missing"}:
                result.errors.append(
                    FieldOutcome(
                        document=document_id,
                        field=label,
                        predicted=(pred_field or {}).get("value"),
                        expected=gold_value,
                        status=status,
                        flagged=bool((pred_field or {}).get("isHumanReviewRequired")),
                    )
                )

        pred_header = predicted.get("invoice", {})
        for name in HEADER_FIELD_NAMES:
            record(name, name, pred_header.get(name), expected["invoice"].get(name))

        pred_rows = predicted.get("lineItems", [])
        gold_rows = expected.get("lineItems", [])
        for pi, gi in match_line_items(pred_rows, gold_rows):
            pred_row = pred_rows[pi] if pi is not None else {}
            gold_row = gold_rows[gi] if gi is not None else {}
            label = f"lineItems[{pi if pi is not None else 'missing'}]"
            for name in LINE_ITEM_FIELD_NAMES:
                record(f"lineItem.{name}", f"{label}.{name}", pred_row.get(name), gold_row.get(name))
    return result


def build_report(result: Evaluation) -> dict[str, Any]:
    return {
        "definition": "missedReviewRatePct = incorrect values not flagged for review / all extracted (non-null) values",
        "overall": result.total.as_dict(),
        "byField": {name: tally.as_dict() for name, tally in sorted(result.by_field.items())},
        "byDocument": {name: tally.as_dict() for name, tally in sorted(result.by_document.items())},
        "errors": [outcome.__dict__ for outcome in result.errors],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score invoice JSON output against an answer key.")
    parser.add_argument("--gold", type=Path, default=Path("eval/gold.json"))
    parser.add_argument("--pred", type=Path, default=Path("output"))
    parser.add_argument("--report", type=Path, default=Path("eval/report.json"))
    parser.add_argument(
        "--no-update-summary",
        action="store_true",
        help="Do not add the evaluation numbers to <pred>/summary.json",
    )
    args = parser.parse_args(argv)
    if not args.gold.exists():
        print(f"Answer key not found: {args.gold}", file=sys.stderr)
        return 1

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    report = build_report(evaluate(gold, args.pred))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = args.pred / "summary.json"
    if not args.no_update_summary and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        overall = report["overall"]
        summary["evaluation"] = {
            "answerKey": str(args.gold),
            "extractedValues": overall["extractedValues"],
            "incorrectValues": overall["incorrectValues"],
            "incorrectNotFlagged": overall["incorrectNotFlagged"],
            "missedReviewRatePct": overall["missedReviewRatePct"],
            "precisionPct": overall["precisionPct"],
            "recallPct": overall["recallPct"],
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    overall = report["overall"]
    print(
        f"Extracted values: {overall['extractedValues']}, correct: {overall['correctValues']}, "
        f"incorrect: {overall['incorrectValues']} ({overall['incorrectNotFlagged']} not flagged). "
        f"Missed-review rate: {overall['missedReviewRatePct']}%. "
        f"Precision: {overall['precisionPct']}%, recall: {overall['recallPct']}%."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
