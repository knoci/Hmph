from datetime import UTC, datetime

import pytest


def test_relative_time_uses_source_timezone_and_preserves_day_precision():
    from hmph.temporal import normalize_time

    # The source is Jan 1 locally, even though the server's UTC date is Dec 31.
    source = datetime(2025, 12, 31, 17, 0, tzinfo=UTC)
    interval = normalize_time("昨天", source, "Asia/Hong_Kong")
    assert interval.start == datetime(2025, 12, 30, 16, 0, tzinfo=UTC)
    assert interval.end == datetime(2025, 12, 31, 16, 0, tzinfo=UTC)
    assert interval.precision == "day" and interval.basis == "relative_to_source"


@pytest.mark.parametrize(
    "expression,start,end,precision",
    [
        ("2024年2月", "2024-02-01T00:00:00+00:00", "2024-03-01T00:00:00+00:00", "month"),
        ("去年", "2024-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00", "year"),
        ("上周", "2025-02-17T00:00:00+00:00", "2025-02-24T00:00:00+00:00", "week"),
        ("2024-02-29", "2024-02-29T00:00:00+00:00", "2024-03-01T00:00:00+00:00", "day"),
        ("两天前", "2025-02-25T00:00:00+00:00", "2025-02-26T00:00:00+00:00", "day"),
    ],
)
def test_time_ranges_preserve_calendar_precision(expression, start, end, precision):
    from hmph.temporal import normalize_time

    interval = normalize_time(expression, datetime(2025, 2, 27, 10, tzinfo=UTC))
    assert interval.start.isoformat() == start and interval.end.isoformat() == end
    assert interval.precision == precision


def test_day_interval_handles_dst_without_inventing_a_fixed_24_hour_day():
    from hmph.temporal import normalize_time

    interval = normalize_time("今天", datetime(2025, 3, 9, 20, tzinfo=UTC), "America/New_York")
    assert interval.start == datetime(2025, 3, 9, 5, tzinfo=UTC)
    assert interval.end == datetime(2025, 3, 10, 4, tzinfo=UTC)


@pytest.mark.parametrize("text", ["最近", "周五", "去年夏天", "2024-02-30", None])
def test_ambiguous_or_invalid_time_stays_unresolved(text):
    from hmph.temporal import normalize_time

    interval = normalize_time(text, datetime(2025, 2, 27, 10, tzinfo=UTC))
    assert interval.start is None and interval.end is None and interval.precision == "unknown"


def test_unknown_legacy_source_timezone_is_not_silently_assumed_to_be_utc():
    from hmph.temporal import normalize_time

    interval = normalize_time("昨天", datetime(2025, 2, 27, 10, tzinfo=UTC), None)
    assert interval.start is None and interval.source_timezone is None
