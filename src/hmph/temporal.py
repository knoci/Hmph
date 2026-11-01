"""Conservative referenced-time intervals, anchored to the source, not processing time."""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class TimeInterval:
    start: datetime | None = None
    end: datetime | None = None
    precision: str = "unknown"
    basis: str = "unresolved"
    source_timezone: str | None = "UTC"


def normalize_time(expression, source_time: datetime, source_timezone="UTC"):
    if source_time.tzinfo is None:
        raise ValueError("Source time must be timezone-aware")
    if source_timezone is None:
        return TimeInterval(source_timezone=None)
    zone = ZoneInfo(source_timezone)
    text = (expression or "").strip().lower()
    anchor = source_time.astimezone(zone).date()

    def interval(start, end, precision, basis):
        return TimeInterval(
            datetime.combine(start, time.min, zone).astimezone(UTC),
            datetime.combine(end, time.min, zone).astimezone(UTC),
            precision,
            basis,
            source_timezone,
        )

    day_offsets = {
        "前天": -2,
        "昨天": -1,
        "昨日": -1,
        "今天": 0,
        "今日": 0,
        "明天": 1,
        "明日": 1,
        "后天": 2,
        "yesterday": -1,
        "today": 0,
        "tomorrow": 1,
    }
    core = re.sub(r"(?:上午|下午|晚上|早上)$", "", text)
    if core in day_offsets:
        day = anchor + timedelta(days=day_offsets[core])
        return interval(day, day + timedelta(days=1), "day", "relative_to_source")
    try:
        explicit = re.fullmatch(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?", text)
        if explicit:
            day = date(*(int(v) for v in explicit.groups()))
            return interval(day, day + timedelta(days=1), "day", "explicit")
        explicit = re.fullmatch(r"(\d{4})[-/年](\d{1,2})月?", text)
        if explicit:
            year, month = (int(v) for v in explicit.groups())
            start = date(year, month, 1)
            end = date(year + month // 12, month % 12 + 1, 1)
            return interval(start, end, "month", "explicit")
        explicit = re.fullmatch(r"(\d{4})年?", text)
        if explicit:
            year = int(explicit[1])
            return interval(date(year, 1, 1), date(year + 1, 1, 1), "year", "explicit")
        years = {
            "前年": -2,
            "去年": -1,
            "今年": 0,
            "明年": 1,
            "last year": -1,
            "this year": 0,
            "next year": 1,
        }
        if text in years:
            year = anchor.year + years[text]
            return interval(date(year, 1, 1), date(year + 1, 1, 1), "year", "relative_to_source")
        months = {
            "上个月": -1,
            "上月": -1,
            "本月": 0,
            "这个月": 0,
            "下个月": 1,
            "last month": -1,
            "this month": 0,
            "next month": 1,
        }
        if text in months:
            number = anchor.year * 12 + anchor.month - 1 + months[text]
            start = date(number // 12, number % 12 + 1, 1)
            end = date((number + 1) // 12, (number + 1) % 12 + 1, 1)
            return interval(start, end, "month", "relative_to_source")
        weeks = {
            "上周": -1,
            "上星期": -1,
            "本周": 0,
            "这周": 0,
            "下周": 1,
            "last week": -1,
            "this week": 0,
            "next week": 1,
        }
        if text in weeks:
            start = anchor - timedelta(days=anchor.weekday()) + timedelta(weeks=weeks[text])
            return interval(start, start + timedelta(days=7), "week", "relative_to_source")
        ago = re.fullmatch(r"(\d{1,4}|[一二两三四五六七八九十])天前", text)
        english = re.fullmatch(r"(\d{1,4}) days? ago", text)
        if ago or english:
            value = (ago or english)[1]
            digits = {c: i for i, c in enumerate("零一二三四五六七八九十")}
            digits["两"] = 2
            offset = int(value) if value.isdigit() else digits[value]
            day = anchor - timedelta(days=offset)
            return interval(day, day + timedelta(days=1), "day", "relative_to_source")
    except (ValueError, OverflowError):
        pass
    return TimeInterval(source_timezone=source_timezone)


def query_intervals(query, query_time: datetime, source_timezone="UTC"):
    pattern = (
        r"(?<!\d)\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?|月)?"
        r"|(?<!\d)\d{4}年?|前年|去年|今年|明年|上个月|下个月|本月|上周|本周|下周"
        r"|昨天|今天|明天|(?:last|this|next) (?:year|month|week)|yesterday|today|tomorrow"
    )
    found = [
        normalize_time(match[0], query_time, source_timezone)
        for match in re.finditer(pattern, query, re.I)
    ]
    return tuple(value for value in found[:4] if value.start is not None)
