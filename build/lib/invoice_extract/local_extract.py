"""Label and pattern-based field extraction (rules backend)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from invoice_extract.align import RawFieldExtraction
from invoice_extract.config import HEADER_FIELD_NAMES, LINE_ITEM_FIELD_NAMES
from invoice_extract.render import DocumentRender, Token


LABEL_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "invoiceNumber": [
        re.compile(r"invoice\s*(?:no|number|#)", re.I),
        re.compile(r"inv\.?\s*no", re.I),
        re.compile(r"发票号"),
    ],
    "invoiceDate": [
        re.compile(r"invoice\s*date", re.I),
        re.compile(r"date", re.I),
        re.compile(r"日期"),
        re.compile(r"ngày", re.I),
    ],
    "supplierName": [
        re.compile(r"from|supplier|vendor|seller", re.I),
        re.compile(r"供应商"),
    ],
    "supplierAddress": [
        re.compile(r"supplier\s*address|vendor\s*address", re.I),
        re.compile(r"地址"),
    ],
    "buyerName": [
        re.compile(r"bill\s*to|buyer|customer|sold\s*to", re.I),
        re.compile(r"客户"),
    ],
    "buyerAddress": [
        re.compile(r"buyer\s*address|bill\s*to\s*address|ship\s*to", re.I),
    ],
    "currency": [
        re.compile(r"currency", re.I),
        re.compile(r"币种"),
    ],
    "subtotalAmount": [
        re.compile(r"sub\s*total|subtotal", re.I),
        re.compile(r"小计"),
    ],
    "taxAmount": [
        re.compile(r"tax|vat|gst", re.I),
        re.compile(r"税"),
    ],
    "totalAmount": [
        re.compile(r"grand\s*total|total\s*due|amount\s*due|total", re.I),
        re.compile(r"合计|总计"),
        re.compile(r"tổng", re.I),
    ],
    "paymentTerms": [
        re.compile(r"payment\s*terms|terms", re.I),
    ],
}

AMOUNT_RE = re.compile(r"[\$€£¥₫]?\s*[\d,]+\.?\d*")
DATE_RE = re.compile(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}|\d{1,2}\s+\w+\s+\d{4}")
CURRENCY_RE = re.compile(r"^(USD|EUR|GBP|HKD|SGD|VND|CNY|JPY|\$|€|£|¥|₫)$", re.I)

LINE_HEADER_RE = re.compile(
    r"description|item|qty|quantity|unit\s*price|amount|price",
    re.I,
)


@dataclass
class RulesExtractor:
    low_confidence: float = 0.55

    def extract(self, doc: DocumentRender) -> tuple[dict[str, RawFieldExtraction], list[dict[str, RawFieldExtraction]]]:
        tokens = doc.tokens
        header = {name: self._extract_header_field(name, tokens) for name in HEADER_FIELD_NAMES}
        line_items = self._extract_line_items(tokens)
        return header, line_items

    def _line_groups(self, tokens: list[Token]) -> list[list[Token]]:
        buckets: dict[tuple[int, int], list[Token]] = {}
        for token in tokens:
            key = (token.page, round(token.y / 8))
            buckets.setdefault(key, []).append(token)
        return [sorted(group, key=lambda t: t.x) for group in buckets.values()]

    def _iter_label_spans(self, tokens: list[Token]) -> list[tuple[str, Token]]:
        spans: list[tuple[str, Token]] = []
        for line in self._line_groups(tokens):
            for start in range(len(line)):
                for size in range(1, min(4, len(line) - start + 1)):
                    chunk = line[start : start + size]
                    text = " ".join(t.text for t in chunk).lower()
                    spans.append((text, chunk[-1]))
        return spans

    def _extract_header_field(self, field_name: str, tokens: list[Token]) -> RawFieldExtraction:
        patterns = LABEL_PATTERNS.get(field_name, [])
        for label_text, label_end in self._iter_label_spans(tokens):
            if not any(p.search(label_text) for p in patterns):
                continue
            value_tokens = self._value_tokens_after_label(label_end, tokens)
            if not value_tokens:
                value_tokens = self._value_tokens_below(label_end, tokens)
            if value_tokens:
                value = " ".join(t.text for t in value_tokens)
                conf = self._pattern_confidence(field_name, value)
                return RawFieldExtraction(
                    value=value,
                    token_ids=[t.id for t in value_tokens],
                    confidence=conf,
                )
        guessed = self._guess_field(field_name, tokens)
        if guessed:
            return guessed
        return RawFieldExtraction(value=None, token_ids=[], confidence=None)

    def _pattern_confidence(self, field_name: str, value: str) -> float:
        if field_name.endswith("Amount") and AMOUNT_RE.search(value):
            return 0.8
        if field_name == "invoiceDate" and DATE_RE.search(value):
            return 0.8
        if field_name == "currency" and CURRENCY_RE.search(value.strip()):
            return 0.8
        return self.low_confidence

    def _guess_field(self, field_name: str, tokens: list[Token]) -> RawFieldExtraction | None:
        if field_name == "currency":
            for token in tokens:
                if CURRENCY_RE.match(token.text.strip()):
                    return RawFieldExtraction(value=token.text, token_ids=[token.id], confidence=self.low_confidence)
        if field_name.endswith("Amount"):
            amounts: list[tuple[float, Token]] = []
            for token in tokens:
                m = AMOUNT_RE.search(token.text)
                if m:
                    try:
                        amounts.append((float(m.group().replace(",", "").replace("$", "")), token))
                    except ValueError:
                        continue
            if amounts and field_name == "totalAmount":
                _, token = max(amounts, key=lambda x: x[0])
                return RawFieldExtraction(value=token.text, token_ids=[token.id], confidence=self.low_confidence)
        if field_name == "invoiceDate":
            for token in tokens:
                if DATE_RE.search(token.text):
                    return RawFieldExtraction(value=token.text, token_ids=[token.id], confidence=self.low_confidence)
        return None

    def _value_tokens_after_label(self, label: Token, tokens: list[Token]) -> list[Token]:
        same_line = [
            t
            for t in tokens
            if t.page == label.page and abs(t.y - label.y) <= max(label.height, 8) and t.x > label.x + label.width - 4
        ]
        same_line.sort(key=lambda t: t.x)
        if not same_line:
            return []
        group = [same_line[0]]
        for t in same_line[1:]:
            if t.x - (group[-1].x + group[-1].width) <= 24:
                group.append(t)
            else:
                break
        return group[:8]

    def _value_tokens_below(self, label: Token, tokens: list[Token]) -> list[Token]:
        below = [
            t
            for t in tokens
            if t.page == label.page and label.y + label.height - 4 <= t.y <= label.y + label.height + 40
        ]
        below.sort(key=lambda t: (t.y, t.x))
        if not below:
            return []
        first_y = below[0].y
        row = [t for t in below if abs(t.y - first_y) <= 8]
        row.sort(key=lambda t: t.x)
        return row[:8]

    def _extract_line_items(self, tokens: list[Token]) -> list[dict[str, RawFieldExtraction]]:
        header_idx = None
        for i, token in enumerate(tokens):
            if LINE_HEADER_RE.search(token.text):
                header_idx = i
                break
        if header_idx is None:
            return []

        header_token = tokens[header_idx]
        row_tokens = [
            t
            for t in tokens
            if t.page == header_token.page and t.y > header_token.y + header_token.height
        ]
        if not row_tokens:
            return []

        rows: dict[int, list[Token]] = {}
        for token in row_tokens:
            bucket = round(token.y / 12) * 12
            rows.setdefault(bucket, []).append(token)

        line_items: list[dict[str, RawFieldExtraction]] = []
        for y_key in sorted(rows.keys())[:20]:
            row = sorted(rows[y_key], key=lambda t: t.x)
            if len(row) < 2:
                continue
            parsed = self._row_to_line_item(row)
            if any(parsed[name].value for name in LINE_ITEM_FIELD_NAMES):
                line_items.append(parsed)
        return line_items

    def _row_to_line_item(self, row: list[Token]) -> dict[str, RawFieldExtraction]:
        texts = [t.text for t in row]
        numeric_indices = [i for i, t in enumerate(texts) if AMOUNT_RE.search(t) or re.match(r"^\d+\.?\d*$", t)]
        description_tokens = row[: numeric_indices[0]] if numeric_indices else row[:1]
        amount_token = row[numeric_indices[-1]] if numeric_indices else None
        qty_token = row[numeric_indices[0]] if numeric_indices else None
        unit_token = row[numeric_indices[1]] if len(numeric_indices) > 2 else None

        def raw(tokens: list[Token] | None, fallback: Token | None = None) -> RawFieldExtraction:
            chosen = tokens or ([fallback] if fallback else [])
            if not chosen:
                return RawFieldExtraction(value=None, token_ids=[], confidence=None)
            return RawFieldExtraction(
                value=" ".join(t.text for t in chosen),
                token_ids=[t.id for t in chosen],
                confidence=self.low_confidence,
            )

        return {
            "description": raw(description_tokens),
            "quantity": raw([qty_token] if qty_token else None),
            "unitPrice": raw([unit_token] if unit_token else None),
            "lineAmount": raw([amount_token] if amount_token else None),
            "itemCode": RawFieldExtraction(value=None, token_ids=[], confidence=None),
            "unitOfMeasure": RawFieldExtraction(value=None, token_ids=[], confidence=None),
        }
