"""`app/llm/markers.py` - the structured marker protocol Claude declares
relevance/citation/stat facts through, replacing the free-text heuristics this
session retired. Pure string/regex parsing, no stack needed.
"""

from __future__ import annotations

from app.llm.markers import extract_cited_skus, extract_stat_markers, is_no_match


def test_is_no_match_true_when_marker_leads() -> None:
    assert is_no_match("[[NO_MATCH]] This catalog doesn't carry that.")


def test_is_no_match_tolerates_leading_whitespace() -> None:
    assert is_no_match("   [[NO_MATCH]] Nothing here matches.")


def test_is_no_match_false_when_marker_is_not_first() -> None:
    """The marker must lead - Claude mentioning it isn't the same as declaring
    the answer a no-match from the start."""
    assert not is_no_match("I found some options. [[NO_MATCH]] isn't relevant here.")


def test_is_no_match_false_with_no_marker() -> None:
    assert not is_no_match("I'd recommend the [[SKU:BOOT-WP-10]].")


def test_extract_cited_skus_finds_all_markers_in_order() -> None:
    answer = "Try [[SKU:BOOT-WP-10]] or [[SKU:JACKET-RAIN-99]] instead."
    assert extract_cited_skus(answer) == ["BOOT-WP-10", "JACKET-RAIN-99"]


def test_extract_cited_skus_ignores_bare_text_outside_a_marker() -> None:
    answer = "BOOT-WP-10 mentioned in plain text is not a citation."
    assert extract_cited_skus(answer) == []


def test_extract_cited_skus_empty_with_no_markers() -> None:
    assert extract_cited_skus("Just a plain sentence.") == []


def test_extract_stat_markers_finds_integers_and_decimals() -> None:
    answer = "Count is [[STAT:94]], average is [[STAT:5200.5]]."
    assert extract_stat_markers(answer) == ["94", "5200.5"]


def test_extract_stat_markers_empty_with_no_markers() -> None:
    assert extract_stat_markers("Nothing quantitative here.") == []
