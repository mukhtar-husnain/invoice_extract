"""Label and pattern-based field extraction (rules backend)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from invoice_extract.align import RawFieldExtraction
from invoice_extract.config import AMOUNT_FIELDS, DATE_FIELDS, HEADER_FIELD_NAMES, LINE_ITEM_FIELD_NAMES
from invoice_extract.normalize import normalize_currency, parse_amount, parse_date, strip_accents
from invoice_extract.render import DocumentRender, Token


def _p(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


# Labels are matched against accent-stripped, lowercased text of 1-4 consecutive words on a line.
LABEL_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "invoiceNumber": [
        _p(r"\binvoice\s*(no|number|#|n)\b"), _p(r"\binv\.?\s*no\b"), _p(r"\bdocument\s*no\b"),
        _p(r"\breceipt\s*no\b"), _p(r"^(no|n[o°º])\s*[.:]?$"), _p(r"\bso\s*\(invoice"), _p(r"tax invoice\s*#"),
        re.compile(r"發票號碼|发票号"),
    ],
    "invoiceDate": [
        _p(r"\binvoice\s*date\b"), _p(r"\bdate\s*of\s*issue\b"), _p(r"^date\s*[:.]?$"), _p(r"^ngay\b"),
        re.compile(r"日期"),
    ],
    "dueDate": [_p(r"\bdue\s*date\b"), _p(r"\bpayment\s*due\s*date\b")],
    "supplierAddress": [_p(r"^dia\s*chi\b"), _p(r"\bsupplier\s*address\b")],
    "buyerName": [
        _p(r"\bbill\s*to\b"), _p(r"\bsold\s*to\b"), _p(r"\binvoice\s*to\b"), _p(r"\bcustomer('?s)?\s*name\b"),
        _p(r"\bcompany\s*name\b"), _p(r"\breceived\s*from\b"), _p(r"\bdon\s*vi\b"), _p(r"\bnguoi\s*mua\b"),
    ],
    "buyerTaxId": [_p(r"^tin\b"), _p(r"\bwith\s*tin\b"), _p(r"\btax\s*id\b")],
    "currency": [_p(r"\bcurrency\b"), _p(r"^curr\.?$"), re.compile(r"币种|幣種")],
    "taxAmount": [
        _p(r"\b(vat|gst|tax)\s*amount\b"), _p(r"\bservice\s*tax\b"), _p(r"^(tax|vat|gst)\s*[:.]?$"),
        _p(r"\bless\s*:?\s*vat\b"), _p(r"\bgst\s*@"), _p(r"%\s*gst\b"), _p(r"\bthue\s*gtgt\b"),
    ],
    "totalAmount": [
        _p(r"\bgrand\s*total\b"), _p(r"\btotal\s*amount(\s*due)?\b"), _p(r"\btotal\s*due\b"), _p(r"\bquote\s*total\b"),
        _p(r"\btotal\s*\w*\s*incl"), _p(r"\bnet\s*amount\b"), _p(r"^total\s*[:.]?$"), _p(r"\btong\s*cong\b"),
        re.compile(r"合計|合计|總計|总计"),
    ],
    "amountDue": [_p(r"\bamount\s*due\b"), _p(r"\bbalance\s*due\b"), _p(r"\bpayment\s*due$"), _p(r"\btotal\s*due\b")],
    "paymentTerms": [_p(r"\bpayment\s*terms?\b"), _p(r"\bterms\s*of\s*payment\b"), _p(r"^payment\s*term$")],
    "purchaseOrderNumber": [
        _p(r"\bp\.?\s*o\.?\s*(no|number|num)\b"), _p(r"\bpurchase\s*order\b"), _p(r"\byour\s*ref\b"),
        _p(r"\bpo\s*reference\b"),
    ],
}

EXCLUDE_PATTERNS: dict[str, re.Pattern[str]] = {
    "invoiceDate": _p(r"due|expir|atp|issued:"),
    "totalAmount": _p(r"sub\s*total|excl|product\s*total|total\s*sales|item\s*count"),
    "invoiceNumber": _p(r"cust|order|reg|account|tel|fax|acc|check|po\b"),
    "taxAmount": _p(r"reg|tin|withholding|exempt|vatable|incl|excl|no\b|invoice"),
    "amountDue": _p(r"date"),
}

COMPANY_SUFFIX_RE = _p(
    r"\b(sdn\.?\s*bhd|pte\.?\s*ltd|ltd|limited|inc|corp|company|opc|services|cong\s*ty|llc|gmbh|plc)\b"
)
LEGAL_SUFFIX_END_RE = _p(r"\b(sdn\.?\s*bhd|pte\.?\s*ltd|ltd|limited|inc|corp|opc|llc|gmbh|plc)\.?\s*$")
DIGIT_RE = re.compile(r"\d")
CURRENCY_IN_TEXT_RE = re.compile(r"(HK\$|S\$|US\$|\bRM\b|\bMYR\b|\bSGD\b|\bHKD\b|\bPHP\b|\bVND\b|\bUSD\b|₱|₫)")

LINE_COLUMN_PATTERNS: dict[str, re.Pattern[str]] = {
    "_code": _p(
        r"^(product|item|material)\s*(code|no\.?)$|^s/?no$|^stt$|^no\.?$|^image$|^uom$|^dvt$|^unit$|^curr\.?$|^currency$"
    ),
    "description": _p(r"description|item\s*name|^product$|particulars|ten\s*hang|^item$|nature\s*of\s*service"),
    "quantity": _p(r"^qty|quantity|so\s*luong"),
    "unitPrice": _p(r"^unit\s*(net\s*|base\s*|gross\s*)?(price|cost)|^price$|^rate$|don\s*gia"),
    "lineAmount": _p(r"amount|^total|thanh\s*tien"),
}
LINE_STOP_RE = _p(
    r"sub\s*total|^total|product\s*total|cong\s*tien|vatable|remarks|total\s*sgd|^(gst|vat|tax)\b"
)
LINE_SKIP_RE = _p(r"amount\s*:|^delivery\b|order\s*no|ordered\s*by|attention|^\d\s+\d\s+\d")
AMOUNT_CELL_RE = re.compile(r"^\D{0,4}\d[\d.,\s]*$")


def _fold(text: str) -> str:
    return strip_accents(text).lower()


@dataclass
class RulesExtractor:
    low_confidence: float = 0.55
    label_confidence: float = 0.8

    def extract(self, doc: DocumentRender) -> tuple[dict[str, RawFieldExtraction], list[dict[str, RawFieldExtraction]]]:
        tokens = doc.tokens
        lines = self._line_groups(tokens)
        header = {name: self._extract_header_field(name, lines, tokens) for name in HEADER_FIELD_NAMES}
        if header["supplierName"].value is None:
            header["supplierName"] = self._guess_supplier(lines)
        if header["currency"].value is None:
            header["currency"] = self._guess_currency(tokens)
        if header["amountDue"].value is None and header["totalAmount"].value is None:
            header["totalAmount"] = self._guess_total(tokens)
        self._demote_tax_id_outside_buyer_block(header, tokens)
        line_items = self._extract_line_items(lines)
        return header, line_items

    def _demote_tax_id_outside_buyer_block(self, header: dict[str, RawFieldExtraction], tokens: list[Token]) -> None:
        """A bare 'TIN' label above the buyer block is usually the seller's registration, not the buyer's."""
        tax_id, buyer = header["buyerTaxId"], header["buyerName"]
        if tax_id.value is None:
            return
        by_id = {t.id: t for t in tokens}
        tax_tokens = [by_id[i] for i in tax_id.token_ids if i in by_id]
        buyer_tokens = [by_id[i] for i in buyer.token_ids if i in by_id]
        if not tax_tokens:
            return
        if not buyer_tokens or (
            tax_tokens[0].page == buyer_tokens[0].page
            and max(t.y + t.height for t in tax_tokens) < min(t.y for t in buyer_tokens)
        ):
            tax_id.confidence = min(tax_id.confidence or 1.0, self.low_confidence)

    def _line_groups(self, tokens: list[Token]) -> list[list[Token]]:
        ordered = sorted(tokens, key=lambda t: (t.page, t.y + t.height / 2, t.x))
        lines: list[list[Token]] = []
        for token in ordered:
            center = token.y + token.height / 2
            if lines:
                last = lines[-1]
                ref = last[0]
                if ref.page == token.page and abs((ref.y + ref.height / 2) - center) <= max(6, ref.height * 0.6):
                    last.append(token)
                    continue
            lines.append([token])
        return [sorted(line, key=lambda t: t.x) for line in lines]

    def _label_matches(self, patterns: list[re.Pattern[str]], chunk: list[Token], line: list[Token], start: int) -> bool:
        label = _fold(" ".join(t.text for t in chunk)).strip()
        at_line_start = start == 0 or chunk[0].x - (line[start - 1].x + line[start - 1].width) > 150
        for pattern in patterns:
            if pattern.pattern.startswith("^") and not at_line_start:
                continue
            if pattern.search(label):
                return True
        return False

    def _find_label(self, name: str, line: list[Token], start: int) -> list[Token] | None:
        """Smallest chunk starting at start whose text is a label for name, so values are never part of the label."""
        patterns = LABEL_PATTERNS[name]
        exclude = EXCLUDE_PATTERNS.get(name)
        for size in range(1, 5):
            chunk = line[start : start + size]
            if len(chunk) < size:
                return None
            if self._label_matches(patterns, chunk, line, start):
                following = [t for t in line[start + size : start + size + 2] if not DIGIT_RE.search(t.text)]
                if exclude and exclude.search(_fold(" ".join(t.text for t in chunk + following))):
                    return None
                while len(chunk) > 1 and self._label_matches(patterns, chunk[1:], line, start + 1):
                    chunk = chunk[1:]
                    start += 1
                return self._extend_label_gloss(chunk, line, start)
        return None

    def _extend_label_gloss(self, chunk: list[Token], line: list[Token], start: int) -> list[Token]:
        """Include a translated gloss such as 'So (Invoice No.):' so the value starts after the colon."""
        if chunk[-1].text.endswith(":"):
            return chunk
        following = line[start + len(chunk) : start + len(chunk) + 3]
        for i, token in enumerate(following):
            if token.text.endswith(":"):
                return chunk + following[: i + 1]
            if i == 0 and not token.text.startswith("("):
                break
        return chunk

    def _extract_header_field(self, name: str, lines: list[list[Token]], tokens: list[Token]) -> RawFieldExtraction:
        if name not in LABEL_PATTERNS:
            return RawFieldExtraction(value=None, token_ids=[], confidence=None)
        for line in lines:
            for start in range(len(line)):
                chunk = self._find_label(name, line, start)
                if chunk is None:
                    continue
                value_tokens = self._value_after(chunk, tokens)
                if not self._acceptable(name, value_tokens):
                    value_tokens = self._value_below(chunk, tokens)
                exclude = EXCLUDE_PATTERNS.get(name)
                if exclude and exclude.search(_fold(" ".join(t.text for t in value_tokens))):
                    continue
                if self._acceptable(name, value_tokens):
                    if name in {"buyerName", "supplierName"}:
                        value_tokens = value_tokens + self._name_continuation(value_tokens, tokens)
                    return self._raw(name, value_tokens, self.label_confidence)
        return RawFieldExtraction(value=None, token_ids=[], confidence=None)

    def _name_continuation(self, value_tokens: list[Token], tokens: list[Token]) -> list[Token]:
        """Second line of a company name that wraps, e.g. 'JADE E-SERVICES' / 'SINGAPORE PTD LTD'."""
        if LEGAL_SUFFIX_END_RE.search(_fold(" ".join(t.text for t in value_tokens))):
            return []
        left = min(t.x for t in value_tokens)
        right = max(t.x + t.width for t in value_tokens)
        top = max(t.y for t in value_tokens)
        height = max(t.height for t in value_tokens)
        below = [
            t
            for t in tokens
            if t.page == value_tokens[0].page
            and top + height * 0.6 <= t.y <= top + height * 1.8
            and t.x + t.width >= left - 10
            and t.x <= right + 10
        ]
        below.sort(key=lambda t: t.x)
        if not below:
            return []
        row_left, row_right = below[0].x, below[-1].x + below[-1].width
        aligned = abs(row_left - left) <= 20 or abs(row_right - right) <= 20
        if aligned and COMPANY_SUFFIX_RE.search(_fold(" ".join(t.text for t in below))):
            return below
        return []

    def _value_after(self, label: list[Token], tokens: list[Token]) -> list[Token]:
        """Words to the right of the label that overlap it vertically, up to the first column gap."""
        top = min(t.y for t in label)
        bottom = max(t.y + t.height for t in label)
        right = label[-1].x + label[-1].width
        label_ids = {t.id for t in label}
        candidates = sorted(
            (
                t
                for t in tokens
                if t.page == label[0].page
                and t.id not in label_ids
                and t.x >= right - 2
                and t.y < bottom
                and t.y + t.height > top
            ),
            key=lambda t: t.x,
        )
        return self._take_until_gap(candidates, right, max(t.height for t in label))

    def _take_until_gap(self, ordered: list[Token], start_x: int, height: int) -> list[Token]:
        group: list[Token] = []
        prev_right = start_x
        for token in ordered:
            gap = token.x - prev_right
            if group and gap > max(40, height * 2.5):
                break
            if not group and gap > max(250, height * 12):
                break
            prev_right = token.x + token.width
            if token.text in {":", "-", "：", "#"} and not group:
                continue
            group.append(token)
            if len(group) >= 12:
                break
        return group

    def _value_below(self, label: list[Token], tokens: list[Token]) -> list[Token]:
        """First row of words under the label, starting near the label's left edge, without a leading sub-label."""
        left = label[0].x
        bottom = max(t.y + t.height for t in label)
        height = max(t.height for t in label)
        below = [
            t
            for t in tokens
            if t.page == label[0].page and bottom - 2 <= t.y <= bottom + height * 3 + 20 and t.x + t.width >= left - 20
        ]
        if not below:
            return []
        label_right = max(t.x + t.width for t in label)
        near_left = [t for t in below if t.x <= label_right + max(200, height * 6)]
        if not near_left:
            return []
        first_y = min(t.y for t in near_left)
        row = sorted(
            (t for t in tokens if t.page == label[0].page and abs(t.y - first_y) <= max(6, height * 0.6)),
            key=lambda t: t.x,
        )
        anchor = min(range(len(row)), key=lambda i: abs(row[i].x - left))
        if row[anchor].x > label_right + max(200, height * 6) or row[anchor].x + row[anchor].width < left - 300:
            return []
        start = anchor
        while start > 0 and row[start].x - (row[start - 1].x + row[start - 1].width) <= max(40, height * 2.5):
            start -= 1
        group = self._take_until_gap(row[start:], row[start].x, height)
        for i, token in enumerate(group[:4]):
            if token.text.endswith(":") and i + 1 < len(group):
                return group[i + 1 :]
        return group

    def _acceptable(self, name: str, value_tokens: list[Token]) -> bool:
        if not value_tokens:
            return False
        text = " ".join(t.text for t in value_tokens)
        if name in AMOUNT_FIELDS:
            return parse_amount(text) is not None
        if name in DATE_FIELDS:
            return parse_date(text) is not None
        if name in {"invoiceNumber", "purchaseOrderNumber", "buyerTaxId"}:
            return bool(DIGIT_RE.search(text))
        if name == "currency":
            return normalize_currency(text) is not None
        return len(text.strip(" :.-")) >= 2

    def _raw(self, name: str, value_tokens: list[Token], confidence: float) -> RawFieldExtraction:
        if name in AMOUNT_FIELDS or name in DATE_FIELDS or name in {"invoiceNumber", "purchaseOrderNumber", "buyerTaxId"}:
            value_tokens = self._trim_to_value(name, value_tokens)
        value = " ".join(t.text for t in value_tokens).strip(" :")
        return RawFieldExtraction(value=value, token_ids=[t.id for t in value_tokens], confidence=confidence)

    def _trim_to_value(self, name: str, value_tokens: list[Token]) -> list[Token]:
        if name in AMOUNT_FIELDS:
            for i, token in enumerate(value_tokens):
                if parse_amount(token.text) is not None:
                    keep = [token]
                    if i > 0 and normalize_currency(value_tokens[i - 1].text):
                        keep.insert(0, value_tokens[i - 1])
                    return keep
        if name in {"invoiceNumber", "purchaseOrderNumber", "buyerTaxId"}:
            first = next((i for i, t in enumerate(value_tokens) if DIGIT_RE.search(t.text)), None)
            if first is None:
                return value_tokens[:1]
            out = [value_tokens[first]]
            if first > 0 and re.fullmatch(r"[A-Z]{2,5}", value_tokens[first - 1].text):
                out.insert(0, value_tokens[first - 1])
            for i in range(first + 1, len(value_tokens)):
                token = value_tokens[i]
                connector = re.fullmatch(r"[-–#/]+", token.text)
                if DIGIT_RE.search(token.text) and not re.fullmatch(r"[A-Za-z]+", token.text):
                    out.append(token)
                elif connector and i + 1 < len(value_tokens) and DIGIT_RE.search(value_tokens[i + 1].text):
                    out.append(token)
                else:
                    break
            return out
        if name in DATE_FIELDS:
            for start in range(len(value_tokens)):
                for size in range(1, 9):
                    window = value_tokens[start : start + size]
                    if len(window) == size and parse_date(" ".join(t.text for t in window)) is not None:
                        return window
        return value_tokens

    def _guess_supplier(self, lines: list[list[Token]]) -> RawFieldExtraction:
        buyer_label = re.compile("|".join(p.pattern for p in LABEL_PATTERNS["buyerName"]), re.I)
        for line in lines[:40]:
            if line[0].page != 1:
                break
            text = " ".join(t.text for t in line)
            folded = _fold(text)
            if buyer_label.search(folded):
                continue
            if COMPANY_SUFFIX_RE.search(folded) and len(text) <= 80:
                return RawFieldExtraction(value=text, token_ids=[t.id for t in line], confidence=self.low_confidence)
        return RawFieldExtraction(value=None, token_ids=[], confidence=None)

    def _guess_currency(self, tokens: list[Token]) -> RawFieldExtraction:
        for token in tokens:
            match = CURRENCY_IN_TEXT_RE.search(token.text)
            if match:
                return RawFieldExtraction(value=match.group(1), token_ids=[token.id], confidence=self.low_confidence)
        return RawFieldExtraction(value=None, token_ids=[], confidence=None)

    def _guess_total(self, tokens: list[Token]) -> RawFieldExtraction:
        best: tuple[float, Token] | None = None
        for token in tokens:
            if not re.search(r"\d[.,]\d{2}$", token.text):
                continue
            amount = parse_amount(token.text)
            if amount is not None and (best is None or amount > best[0]):
                best = (amount, token)
        if best is None:
            return RawFieldExtraction(value=None, token_ids=[], confidence=None)
        return RawFieldExtraction(value=best[1].text, token_ids=[best[1].id], confidence=self.low_confidence)

    def _find_table_header(self, lines: list[list[Token]]) -> tuple[int, dict[str, tuple[int, int]]] | None:
        preferred_words = {"description": "description", "unitPrice": "net", "lineAmount": "amount"}
        for idx, line in enumerate(lines):
            columns: dict[str, tuple[int, int]] = {}
            start = 0
            while start < len(line):
                matched = 0
                for size in (3, 2, 1):
                    chunk = line[start : start + size]
                    if len(chunk) < size:
                        continue
                    if size > 1 and any(
                        b.x - (a.x + a.width) > max(30, a.height * 2) for a, b in zip(chunk, chunk[1:])
                    ):
                        continue
                    label = _fold(" ".join(t.text for t in chunk)).strip(" :()")
                    for col, pattern in LINE_COLUMN_PATTERNS.items():
                        if not pattern.search(label):
                            continue
                        span = (chunk[0].x, chunk[-1].x + chunk[-1].width)
                        if col in columns:
                            spare = f"_{col}{len(columns)}"
                            if preferred_words.get(col, "\0") in label:
                                columns[spare], columns[col] = columns[col], span
                            else:
                                columns[spare] = span
                        else:
                            columns[col] = span
                        matched = size
                        break
                    if matched:
                        break
                start += matched or 1
            real = {c for c in columns if not c.startswith("_")}
            if "lineAmount" in real and len(real) >= 2:
                return idx, columns
        return None

    @staticmethod
    def _column_for(token: Token, columns: dict[str, tuple[int, int]]) -> str:
        """Column whose header overlaps the word most; nearest header edge when nothing overlaps."""
        x1, x2 = token.x, token.x + token.width

        def score(span: tuple[int, int]) -> tuple[float, float]:
            overlap = min(x2, span[1]) - max(x1, span[0])
            distance = 0.0 if overlap > 0 else min(abs(x1 - span[1]), abs(span[0] - x2))
            return (-max(overlap, 0), distance)

        return min(columns, key=lambda c: score(columns[c]))

    def _extract_line_items(self, lines: list[list[Token]]) -> list[dict[str, RawFieldExtraction]]:
        found = self._find_table_header(lines)
        if found is None:
            return []
        header_idx, columns = found
        page = lines[header_idx][0].page
        items: list[dict[str, RawFieldExtraction]] = []
        last_desc: list[Token] = []
        last_bottom = 0.0
        for line in lines[header_idx + 1 :]:
            if line[0].page != page:
                break
            text = _fold(" ".join(t.text for t in line)).strip()
            if LINE_STOP_RE.search(text):
                break
            if last_desc and self._continues_description(line, last_desc, last_bottom, columns):
                last_desc.extend(line)
                desc = items[-1]["description"]
                desc.value = " ".join(t.text for t in last_desc)
                desc.token_ids = [t.id for t in last_desc]
                last_bottom = max(t.y + t.height for t in line)
                continue
            last_desc = []
            if LINE_SKIP_RE.search(text):
                continue
            cells: dict[str, list[Token]] = {}
            for token in line:
                column = self._column_for(token, columns)
                cell = cells.setdefault(column, [])
                is_word = not DIGIT_RE.search(token.text)
                unit_suffix = bool(cell) and DIGIT_RE.search(cell[-1].text) and len(token.text) <= 4
                if column in AMOUNT_FIELDS and is_word and not unit_suffix and normalize_currency(token.text) is None:
                    column = "description"
                    cell = cells.setdefault(column, [])
                cell.append(token)
            if cells.get("description"):
                cells["description"].sort(key=lambda t: t.x)
            amount_text = " ".join(t.text for t in cells.get("lineAmount", []))
            if not AMOUNT_CELL_RE.match(amount_text) or parse_amount(amount_text) is None:
                continue
            if not cells.get("description"):
                continue
            row: dict[str, RawFieldExtraction] = {}
            for name in LINE_ITEM_FIELD_NAMES:
                toks = cells.get(name, [])
                confidence = self.label_confidence
                if name == "description" and not toks:
                    toks = [t for t in line if parse_amount(t.text) is None][:12]
                    confidence = self.low_confidence
                row[name] = (
                    RawFieldExtraction(
                        value=" ".join(t.text for t in toks), token_ids=[t.id for t in toks], confidence=confidence
                    )
                    if toks
                    else RawFieldExtraction(value=None, token_ids=[], confidence=None)
                )
            items.append(row)
            if cells.get("description") and row["description"].confidence == self.label_confidence:
                last_desc = list(cells["description"])
                last_bottom = max(t.y + t.height for t in line)
            if len(items) >= 60:
                break
        return items

    def _continues_description(
        self, line: list[Token], last_desc: list[Token], last_bottom: float, columns: dict
    ) -> bool:
        """A wrapped description: next line, same left edge, nothing in any other column."""
        height = max(t.height for t in last_desc)
        if line[0].y - last_bottom > height * 0.8:
            return False
        if abs(line[0].x - min(t.x for t in last_desc)) > max(20, height):
            return False
        return all(self._column_for(t, columns) == "description" and not DIGIT_RE.fullmatch(t.text) for t in line)
