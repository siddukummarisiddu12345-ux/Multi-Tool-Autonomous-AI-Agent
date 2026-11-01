"""Secret redaction and safe error text."""
from __future__ import annotations

import re
from typing import Iterable

_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)(api[_-]?key=)[^&\s]+"),
]


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    for s in secrets:
        if s and len(s) >= 6:
            text = text.replace(s, "[REDACTED]")
    for p in _PATTERNS:
        text = p.sub(lambda m: (m.group(1) + "[REDACTED]") if m.groups() else "[REDACTED]", text)
    return text


def safe_error(exc: BaseException, secrets: Iterable[str] = (), limit: int = 200) -> str:
    detail = redact(str(exc), secrets).replace("\n", " ").strip()
    name = type(exc).__name__
    return truncate(f"{name}: {detail}" if detail else name, limit)
