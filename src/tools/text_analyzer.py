"""Text statistics and keyword frequencies."""
from __future__ import annotations

import re
from collections import Counter

from src.tools.base import ToolError

MAX_CHARS = 100_000
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is it its me my no not of on "
    "or our she so than that the their them then there these they this to was we were what when which who "
    "will with you your about after all also any been can could do does did had how just more most only "
    "other out over some such too very would".split()
)
_WORD = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)


def analyze_text(text, top_n: int = 10) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise ToolError("Text is empty.")
    if len(text) > MAX_CHARS:
        raise ToolError(f"Text too long (limit {MAX_CHARS:,} characters).")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 50:
        raise ToolError("top_n must be an integer between 1 and 50.")
    words = _WORD.findall(text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if _WORD.search(s)]
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    lowered = [w.lower() for w in words]
    keywords = Counter(w for w in lowered if len(w) >= 3 and w not in STOPWORDS and not w.isdigit())
    return {
        "word_count": len(words),
        "character_count": len(text),
        "character_count_no_spaces": len(re.sub(r"\s", "", text)),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "unique_words": len(set(lowered)),
        "keywords": [{"word": w, "count": c} for w, c in keywords.most_common(top_n)],
    }
