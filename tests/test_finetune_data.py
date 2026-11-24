from fine_tune.prepare_data import clean_record, dedupe


def record(user: str, reply: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are Muffin."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply},
        ]
    }


def test_clean_record_keeps_short_khmerlish():
    cleaned = clean_record(record("Pong tvr ey", "Pg nham bay"), 200, 1)
    assert cleaned is not None
    assert cleaned["messages"][-1]["content"] == "Pg nham bay"


def test_clean_record_drops_urls_and_rambles():
    assert clean_record(record("look", "https://example.com"), 200, 1) is None
    assert clean_record(record("hey", "word " * 100), 200, 1) is None
    assert clean_record(record("hey", "Bajjjj"), 200, 1) is None


def test_dedupe_removes_identical_exchanges():
    rows = [record("hi", "Bart"), record("HI", "bart"), record("hi", "Ort")]
    kept, dropped = dedupe(rows)
    assert len(kept) == 2
    assert dropped == 1
