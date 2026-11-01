import pytest

from src.tools.base import ToolError
from src.tools.text_analyzer import analyze_text

TEXT = "Python agents use tools. Tools return results! Do agents explain results?\n\nSecond paragraph, agents again."


def test_counts():
    r = analyze_text(TEXT)
    assert r["sentence_count"] == 4 and r["paragraph_count"] == 2
    assert r["word_count"] == 15 and r["character_count"] == len(TEXT)
    assert r["character_count_no_spaces"] == len("".join(TEXT.split()))


def test_keyword_frequencies():
    r = analyze_text(TEXT, top_n=3)
    top = {k["word"]: k["count"] for k in r["keywords"]}
    assert top["agents"] == 3 and len(r["keywords"]) == 3
    assert "the" not in top and "do" not in top


def test_single_word_and_unicode():
    r = analyze_text("héllo")
    assert r["word_count"] == 1 and r["sentence_count"] == 1 and r["paragraph_count"] == 1


def test_invalid_inputs():
    for bad in ["", "   \n ", None, 5]:
        with pytest.raises(ToolError):
            analyze_text(bad)
    with pytest.raises(ToolError):
        analyze_text("x" * 100_001)
    for n in (0, 51, True, "3"):
        with pytest.raises(ToolError):
            analyze_text("hello world", top_n=n)
