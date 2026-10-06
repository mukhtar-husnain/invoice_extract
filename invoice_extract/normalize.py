"""Parse printed amounts, dates, rates, and currencies into comparable values."""

from __future__ import annotations

import re
import unicodedata
from datetime import date

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

CURRENCY_ALIASES = {
    "MYR": "MYR", "RM": "MYR",
    "SGD": "SGD", "S$": "SGD",
    "HKD": "HKD", "HK$": "HKD",
    "PHP": "PHP", "₱": "PHP", "P": "PHP", "PESO": "PHP", "PESOS": "PHP",
    "VND": "VND", "₫": "VND", "ĐỒNG": "VND", "DONG": "VND", "Đ": "VND",
    "USD": "USD", "US$": "USD", "$": "USD",
    "EUR": "EUR", "€": "EUR",
    "GBP": "GBP", "£": "GBP",
    "CNY": "CNY", "RMB": "CNY", "¥": "CNY",
    "JPY": "JPY", "AUD": "AUD", "IDR": "IDR", "THB": "THB", "INR": "INR",
}

_CURRENCY_TOKEN_RE = re.compile(
    r"(HK\$|S\$|US\$|RM|MYR|SGD|HKD|PHP|VND|USD|EUR|GBP|CNY|RMB|JPY|AUD|IDR|THB|INR|₱|₫|€|£|¥|\$)",
    re.I,
)


def parse_amount(text: str | None) -> float | None:
    """Parse '1,159.30', '39.254.545', 'S$ 623.48', 'MYR 400.00', '4 Btl' into a float."""
    if not text:
        return None
    match = re.search(r"-?\d[\d.,\s]*", text)
    if not match:
        return None
    group = match.group().strip()
    if re.fullmatch(r"-?\d+\s\d{2}", group):
        group = group.replace(" ", ".")
    raw = re.sub(r"\s", "", group).rstrip(".,")
    if not raw or raw == "-":
        return None
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif raw.count(".") > 1:
        raw = raw.replace(".", "")
    elif raw.count(",") > 1:
        raw = raw.replace(",", "")
    elif "," in raw:
        head, tail = raw.split(",")
        raw = head + tail if len(tail) == 3 else head + "." + tail
    try:
        return float(raw)
    except ValueError:
        return None


_ID_GROUPS_RE = re.compile(r"\d+\s*-\s*\d+\s*-\s*\d+")
_DATE_LIKE_RE = re.compile(r"\d{1,4}\s*[/.]\s*\d{1,2}\s*[/.]\s*\d{2,4}(?!\d)")
_LETTER_DIGIT_RE = re.compile(r"[^\W\d_]\d|\d[^\W\d_]")
_UNIT_SUFFIX_RE = re.compile(r"^\s*[^\W\d_]{1,6}\d{0,4}\.?\s*$")


def looks_like_amount(text: str, *, allow_unit: bool = False) -> bool:
    """A money amount or quantity, not a percentage, date, tax id, or reference number."""
    if parse_amount(text) is None or "%" in text:
        return False
    if _ID_GROUPS_RE.search(text) or _DATE_LIKE_RE.search(text):
        return False
    body = _CURRENCY_TOKEN_RE.sub(" ", text)
    if _LETTER_DIGIT_RE.search(body) and not allow_unit:
        return False
    rest = re.sub(r"[-+(]?\d[\d.,\s]*\)?", " ", body, count=1).strip(" :")
    if not rest or rest in {"-", "/-"} or normalize_currency(rest) is not None:
        return True
    return allow_unit and bool(_UNIT_SUFFIX_RE.match(rest))


_NAME_CHARS_RE = re.compile(r"[^\w\s.,&'()\-/#@:+\"*%’‘“”–—\[\]×!?;]")


def looks_like_text(text: str, *, min_letter_ratio: float = 0.6) -> bool:
    """Readable name or description: mostly letters, no stray OCR symbols such as '€', '§', or '\\'."""
    if _NAME_CHARS_RE.search(text.replace("\u200b", "")):
        return False
    chars = [c for c in text if c.isalnum()]
    return bool(chars) and sum(c.isalpha() for c in chars) / len(chars) >= min_letter_ratio


def parse_rate(text: str | None) -> float | None:
    """Parse '10%' or '@09%' into a percentage number."""
    if not text:
        return None
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
    if match:
        return float(match.group(1).replace(",", "."))
    return parse_amount(text)


def _year(value: int) -> int:
    return value + 2000 if value < 100 else value


def parse_date(text: str | None) -> date | None:
    """Day-first parse of printed dates, including '02ND FEB 2026' and Vietnamese 'ngày 22 tháng 01 năm 2026'."""
    if not text:
        return None
    lowered = strip_accents(text).lower()
    lowered = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", lowered)

    numeric = re.search(r"(\d{1,4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,4})", lowered)
    if numeric:
        a, b, c = (int(x) for x in numeric.groups())
        try:
            if a > 31:
                return date(a, b, c)
            if b > 12 and a <= 12:
                return date(_year(c), a, b)
            return date(_year(c), b, a)
        except ValueError:
            return None

    month = None
    for word in re.findall(r"[a-z]+", lowered):
        if word in MONTHS:
            month = MONTHS[word]
            break
    numbers = [int(x) for x in re.findall(r"\d+", lowered)]
    if month is not None and numbers:
        years = [n for n in numbers if n >= 1900]
        days = [n for n in numbers if 1 <= n <= 31]
        year = years[0] if years else (_year(numbers[-1]) if len(numbers) >= 2 else None)
        day = days[0] if days else None
        if year and day:
            try:
                return date(year, month, day)
            except ValueError:
                return None

    if "thang" in lowered and len(numbers) >= 3:
        day, mon, year = numbers[-3], numbers[-2], numbers[-1]
        try:
            return date(_year(year), mon, day)
        except ValueError:
            return None
    return None


def normalize_currency(text: str | None) -> str | None:
    """Map 'RM', 'S$', 'MYR 400.00', 'pesos' to an ISO code."""
    if not text:
        return None
    stripped = text.strip().upper()
    if stripped in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[stripped]
    match = _CURRENCY_TOKEN_RE.search(text)
    if match:
        return CURRENCY_ALIASES.get(match.group(1).upper())
    for word in re.findall(r"[^\W\d_]+", stripped):
        if word in CURRENCY_ALIASES and len(word) > 1:
            return CURRENCY_ALIASES[word]
    return None


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_text(text: str | None) -> str:
    """Casefold, drop accents and punctuation, collapse whitespace."""
    if not text:
        return ""
    cleaned = strip_accents(text).casefold()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_identifier(text: str | None) -> str:
    """Keep only letters and digits, uppercased: 'CSOS 202602-931' -> 'CSOS202602931'."""
    if not text:
        return ""
    return re.sub(r"[\W_]", "", strip_accents(text)).upper()
