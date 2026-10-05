from __future__ import annotations

import pytest

from invoice_extract.render import Token


@pytest.fixture
def sample_tokens() -> list[Token]:
    return [
        Token(id=0, text="Invoice", page=1, x=10, y=10, width=50, height=12, confidence=1.0),
        Token(id=1, text="No:", page=1, x=70, y=10, width=20, height=12, confidence=1.0),
        Token(id=2, text="INV-001", page=1, x=100, y=10, width=60, height=12, confidence=1.0),
        Token(id=3, text="Total", page=1, x=10, y=100, width=40, height=12, confidence=1.0),
        Token(id=4, text="$123.45", page=1, x=60, y=100, width=50, height=12, confidence=1.0),
        Token(id=5, text="Qty", page=1, x=10, y=140, width=30, height=12, confidence=0.9),
        Token(id=6, text="2", page=1, x=50, y=140, width=10, height=12, confidence=0.9),
        Token(id=7, text="10.00", page=1, x=70, y=140, width=40, height=12, confidence=0.9),
        Token(id=8, text="20.00", page=1, x=120, y=140, width=40, height=12, confidence=0.9),
    ]
