"""Unit tests for follow-up question normalisation (no network)."""

from langchain_zerogpu.tools import _questions


def test_parses_the_json_array_the_model_returns() -> None:
    raw = '["What are the benefits of EVs?","How do they compare to hybrids?"]'
    assert _questions(raw) == [
        "What are the benefits of EVs?",
        "How do they compare to hybrids?",
    ]


def test_falls_back_to_newline_delimited_output() -> None:
    raw = "How long does an EV battery last?\n\nIs the grid ready?\n"
    assert _questions(raw) == [
        "How long does an EV battery last?",
        "Is the grid ready?",
    ]


def test_drops_blank_entries() -> None:
    assert _questions('["Real question?", "  ", ""]') == ["Real question?"]
