"""Current date/time and date-difference calculations."""
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Callable, Optional

from src.tools.base import ToolError


def _parse(value: Optional[str], label: str):
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"'{label}' is required (ISO format, e.g. 2025-03-01 or 2025-03-01T09:30:00).")
    value = value.strip()
    try:
        if len(value) == 10:
            return date.fromisoformat(value)
        return datetime.fromisoformat(value)
    except ValueError:
        raise ToolError(f"'{label}' is not a valid ISO date/datetime: {value[:40]!r}.") from None


def _add_months(d: date, months: int) -> date:
    idx = d.year * 12 + (d.month - 1) + months
    y, m = divmod(idx, 12)
    return date(y, m + 1, min(d.day, calendar.monthrange(y, m + 1)[1]))


def _ymd(a: date, b: date) -> tuple[int, int, int]:
    """Calendar difference between a <= b as (years, months, days)."""
    months = (b.year - a.year) * 12 + b.month - a.month
    if b.day < a.day:
        months -= 1
    anchor = _add_months(a, months)
    return months // 12, months % 12, (b - anchor).days


def _now(timezone: Optional[str], now: Optional[Callable[[], datetime]]) -> dict:
    if timezone:
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(timezone)
        except Exception:  # unknown key, bad format, or tzdata missing
            raise ToolError(f"Unknown timezone '{timezone[:40]}'. Use an IANA name like 'Asia/Kolkata'.") from None
        current = (now() if now else datetime.now(tz)).astimezone(tz)
    else:
        current = (now() if now else datetime.now()).astimezone()
    return {
        "operation": "now", "iso": current.isoformat(timespec="seconds"),
        "date": current.date().isoformat(), "time": current.strftime("%H:%M:%S"),
        "weekday": current.strftime("%A"), "timezone": current.tzname() or "",
        "utc_offset": current.strftime("%z"), "unix_timestamp": int(current.timestamp()),
    }


def _diff(start, end) -> dict:
    a, b = _parse(start, "start"), _parse(end, "end")
    both_dates = type(a) is date and type(b) is date
    if not both_dates:
        a = a if isinstance(a, datetime) else datetime(a.year, a.month, a.day, tzinfo=getattr(b, "tzinfo", None))
        b = b if isinstance(b, datetime) else datetime(b.year, b.month, b.day, tzinfo=a.tzinfo)
        if (a.tzinfo is None) != (b.tzinfo is None):
            raise ToolError("Cannot mix timezone-aware and naive datetimes.")
    first, last = (a, b) if a <= b else (b, a)
    delta = last - first
    total = int(delta.total_seconds())
    y, m, d = _ymd(first.date() if isinstance(first, datetime) else first,
                   last.date() if isinstance(last, datetime) else last)
    out = {
        "operation": "diff", "start": str(start).strip(), "end": str(end).strip(),
        "direction": "end is after start" if a <= b else "end is before start",
        "total_days": delta.days, "total_hours": round(total / 3600, 4),
        "total_seconds": total, "weeks": delta.days // 7, "extra_days": delta.days % 7,
        "calendar_difference": {"years": y, "months": m, "days": d},
    }
    return out


def datetime_info(operation, timezone=None, start=None, end=None,
                  now: Optional[Callable[[], datetime]] = None) -> dict:
    if operation == "now":
        return _now(timezone, now)
    if operation == "diff":
        return _diff(start, end)
    raise ToolError("operation must be 'now' or 'diff'.")
