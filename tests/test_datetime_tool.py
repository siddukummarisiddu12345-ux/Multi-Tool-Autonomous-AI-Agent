from datetime import datetime, timezone

import pytest

from src.tools.base import ToolError
from src.tools.datetime_tool import datetime_info

FIXED = lambda: datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)


def test_now_with_timezone():
    r = datetime_info("now", timezone="UTC", now=FIXED)
    assert r["iso"] == "2025-01-15T10:30:00+00:00" and r["weekday"] == "Wednesday"
    assert r["date"] == "2025-01-15" and r["unix_timestamp"] == 1736937000


def test_now_other_timezone():
    r = datetime_info("now", timezone="Asia/Kolkata", now=FIXED)
    assert r["time"] == "16:00:00" and r["utc_offset"] == "+0530"


def test_now_system_clock():
    r = datetime_info("now")
    assert r["operation"] == "now" and len(r["date"]) == 10


def test_diff_dates():
    r = datetime_info("diff", start="2025-01-01", end="2025-03-15")
    assert r["total_days"] == 73 and r["weeks"] == 10 and r["extra_days"] == 3
    assert r["calendar_difference"] == {"years": 0, "months": 2, "days": 14}
    assert r["direction"] == "end is after start"


def test_diff_reversed_and_month_end():
    r = datetime_info("diff", start="2025-03-15", end="2025-01-01")
    assert r["total_days"] == 73 and r["direction"] == "end is before start"
    assert datetime_info("diff", start="2024-01-31", end="2024-03-01")["calendar_difference"] == \
        {"years": 0, "months": 1, "days": 1}
    assert datetime_info("diff", start="2024-02-28", end="2024-03-01")["total_days"] == 2  # leap year


def test_diff_datetimes():
    r = datetime_info("diff", start="2025-01-01T00:00:00", end="2025-01-02T12:00:00")
    assert r["total_hours"] == 36.0 and r["total_days"] == 1
    r = datetime_info("diff", start="2025-01-01", end="2025-01-01T06:00:00")
    assert r["total_seconds"] == 21600


def test_invalid_inputs():
    bad_calls = [
        dict(operation="later"), dict(operation="diff"), dict(operation="diff", start="2025-01-01"),
        dict(operation="diff", start="2025-13-45", end="2025-01-01"),
        dict(operation="diff", start="nonsense", end="2025-01-01"),
        dict(operation="now", timezone="Mars/Base"),
        dict(operation="diff", start="2025-01-01T00:00:00", end="2025-01-01T00:00:00+00:00"),
    ]
    for kw in bad_calls:
        with pytest.raises(ToolError):
            datetime_info(**kw)
