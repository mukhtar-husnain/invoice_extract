from invoice_extract.config import PipelineConfig
from invoice_extract.evaluate import match_line_items, score_field, values_match


def _field(value, flagged=False, bbox=True):
    return {
        "value": value,
        "bbox": {"page": 1, "x1": 0, "y1": 0, "x2": 1, "y2": 1} if bbox else None,
        "isHumanReviewRequired": flagged,
        "reviewReasons": ["LOW_CONFIDENCE"] if flagged else [],
    }


def test_values_match_normalizes_by_field_type():
    assert values_match("totalAmount", "RM 1,234.50", "1234.5")
    assert values_match("totalAmount", "39.254.545", "39254545")
    assert values_match("invoiceDate", "02/03/2024", "2 March 2024")
    assert values_match("currency", "S$", "SGD")
    assert values_match("invoiceNumber", "INV-0012", "inv 0012")
    assert not values_match("totalAmount", "100.00", "101.00")


def test_unflagged_wrong_value_counts_as_missed_review():
    tally, status = score_field("totalAmount", _field("99.00"), "100.00")
    assert status == "incorrect_not_flagged"
    assert tally.as_dict()["missedReviewRatePct"] == 100.0


def test_flagged_wrong_value_is_not_a_missed_review():
    tally, status = score_field("totalAmount", _field("99.00", flagged=True), "100.00")
    assert status == "incorrect_flagged"
    assert tally.incorrect_not_flagged == 0


def test_empty_value_is_never_counted_as_extracted():
    tally, status = score_field("dueDate", _field(None, bbox=False), None)
    assert status == "correct_empty"
    assert tally.extracted == 0
    tally, status = score_field("dueDate", _field(None, bbox=False), "01/01/2024")
    assert status == "missing"
    assert tally.extracted == 0


def test_answer_key_list_accepts_any_option():
    _, status = score_field("totalAmount", _field("500"), ["450", "500"])
    assert status == "correct"


def test_match_line_items_pairs_by_description_and_amount():
    pred = [
        {"description": _field("Widget B"), "lineAmount": _field("20.00")},
        {"description": _field("Widget A"), "lineAmount": _field("10.00")},
    ]
    gold = [{"description": "Widget A", "lineAmount": "10.00"}, {"description": "Widget B", "lineAmount": "20.00"}]
    assert sorted(p for p in match_line_items(pred, gold) if None not in p) == [(0, 1), (1, 0)]


def test_blank_env_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("INVOICE_OLLAMA_MODEL", "")
    monkeypatch.setenv("INVOICE_OLLAMA_NUM_CTX", " ")
    config = PipelineConfig.from_env()
    assert config.ollama_model == PipelineConfig.ollama_model
    assert config.ollama_num_ctx == PipelineConfig.ollama_num_ctx
