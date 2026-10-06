from __future__ import annotations

from datetime import date

import pytest

from invoice_extract.normalize import (
    looks_like_amount,
    looks_like_text,
    normalize_currency,
    parse_amount,
    parse_date,
    parse_rate,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("13,764.48", 13764.48),
        ("39.254.545", 39254545.0),
        ("S$ 623.48", 623.48),
        ("MYR 400.00", 400.0),
        ("P 15,000", 15000.0),
        ("4 Btl", 4.0),
        ("1.20", 1.2),
        ("1.234,56", 1234.56),
        ("n/a", None),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("02ND FEB 2026", date(2026, 2, 2)),
        ("25/02/26", date(2026, 2, 25)),
        ("03.09.2025", date(2025, 9, 3)),
        ("27 FEBRUARY 2026", date(2026, 2, 27)),
        ("November 23, 2025", date(2025, 11, 23)),
        ("Ngày (day) 22 tháng (month) 01 năm (year) 2026", date(2026, 1, 22)),
        ("2026-01-12", date(2026, 1, 12)),
        ("not a date", None),
    ],
)
def test_parse_date(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("RM", "MYR"), ("S$", "SGD"), ("MYR 400.00", "MYR"), ("HKD", "HKD"), ("pesos", "PHP"), ("xyz", None)],
)
def test_normalize_currency(text, expected):
    assert normalize_currency(text) == expected


def test_parse_rate():
    assert parse_rate("10%") == 10.0
    assert parse_rate("@09%") == 9.0


def test_ocr_dropped_decimal_point():
    assert parse_amount("S$ 240 00") == 240.0
    assert parse_amount("1 000") == 1000.0


@pytest.mark.parametrize(
    ("text", "allow_unit", "expected"),
    [
        ("RM 1,234.50", False, True),
        ("39.254.545 đ", False, True),
        ("(1,000.00)", False, True),
        ("9%", False, False),
        ("143-009-159-00000", False, False),
        ("15/10/25", False, False),
        ("201823456M", False, False),
        ("4 Btl", True, True),
        ("2 PK10", True, True),
        ("750m!", True, False),
    ],
)
def test_looks_like_amount(text, allow_unit, expected):
    assert looks_like_amount(text, allow_unit=allow_unit) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("BF JADE E-SERVICES PHILIPPINES INC.", True),
        ("Công ty TNHH Global Fashion", True),
        ("新達(香港)有限公司", True),
        ("BF JAOE E- €E£v", False),
        ("505387672 OHBHESLO", False),
        ("NAME § SIGN", False),
    ],
)
def test_looks_like_text(text, expected):
    assert looks_like_text(text) is expected
