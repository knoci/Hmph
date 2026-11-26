"""Multi-feature reranking, session-burst-aware diversity, and relative-time arithmetic anchoring."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from hmph.lexical import tokenize

_ISO_DATE = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_EN_DATE = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})",
    re.I,
)
_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_NUM_WORDS = {
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "couple": 2,
    "few": 3,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_REL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\byesterday\b|昨天|昨日", re.I), "day:-1"),
    (re.compile(r"\bday before yesterday\b|前天", re.I), "day:-2"),
    (re.compile(r"\btomorrow\b|明天|明日", re.I), "day:+1"),
    (re.compile(r"\blast week\b|上周|上星期|上礼拜", re.I), "day:-7"),
    (re.compile(r"\bnext week\b|下周|下星期|下礼拜", re.I), "day:+7"),
    (re.compile(r"\blast month\b|上个月|上月", re.I), "day:-30"),
    (re.compile(r"\blast year\b|去年", re.I), "year:-1"),
    (re.compile(r"前年", re.I), "year:-2"),
    (re.compile(r"\bnext year\b|明年", re.I), "year:+1"),
    (
        re.compile(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\s+years?\s+ago\b",
            re.I,
        ),
        "years_ago",
    ),
    (
        re.compile(
            r"\bfor\s+(?:about\s+|around\s+|nearly\s+|over\s+|almost\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\b",
            re.I,
        ),
        "years_ago",
    ),
    (
        re.compile(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\s+months?\s+ago\b",
            re.I,
        ),
        "months_ago",
    ),
    (
        re.compile(
            r"\bfor\s+(?:about\s+|around\s+|nearly\s+|over\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+months?\b",
            re.I,
        ),
        "months_ago",
    ),
    (
        re.compile(
            r"\b(\d+|one|two|three|four|five|six|couple|few)\s+weeks?\s+ago\b",
            re.I,
        ),
        "weeks_ago",
    ),
    (
        re.compile(
            r"\b(\d+|one|two|three|four|five|six|seven|couple|few)\s+days?\s+ago\b",
            re.I,
        ),
        "days_ago",
    ),
    (re.compile(r"(\d+|[一二两三四五六七八九十])\s*年(?:前|了|多)", re.I), "years_ago"),
    (re.compile(r"(\d+|[一二两三四五六七八九十])\s*个?月前", re.I), "months_ago"),
    (re.compile(r"(\d+|[一二两三四五六七八九十])\s*周前", re.I), "weeks_ago"),
    (re.compile(r"(\d+|[一二两三四五六七八九十])\s*天前", re.I), "days_ago"),
]


def parse_anchor_date(timestamp_str: str | None) -> datetime | None:
    if not timestamp_str:
        return None
    m = _ISO_DATE.search(timestamp_str)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    m2 = _EN_DATE.search(timestamp_str)
    if m2:
        try:
            return datetime(int(m2[3]), _MONTH_MAP[m2[2].lower()], int(m2[1]))
        except ValueError:
            return None
    return None


def _resolve_relative_anchors(text: str, recorded_at: str | None) -> list[tuple[str, str]]:
    """Return list of (matched_phrase, resolved_date_or_year_str)."""
    base = parse_anchor_date(recorded_at)
    if base is None:
        return []
    anchors: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pat, kind in _REL_PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(0)
            if kind.startswith("day:"):
                delta = int(kind.split(":")[1])
                resolved = (base + timedelta(days=delta)).strftime("%Y-%m-%d")
            elif kind.startswith("year:"):
                delta = int(kind.split(":")[1])
                resolved = str(base.year + delta)
            else:
                raw_n = m.group(1).lower()
                n = int(raw_n) if raw_n.isdigit() else _NUM_WORDS.get(raw_n, 2)
                if kind == "years_ago":
                    resolved = str(base.year - n)
                elif kind == "months_ago":
                    resolved = (base - timedelta(days=30 * n)).strftime("%Y-%m")
                elif kind == "weeks_ago":
                    resolved = (base - timedelta(days=7 * n)).strftime("%Y-%m-%d")
                else:
                    resolved = (base - timedelta(days=n)).strftime("%Y-%m-%d")
            key = f"{phrase.lower()}={resolved}"
            if key not in seen:
                seen.add(key)
                anchors.append((phrase, resolved))
            if len(anchors) >= 3:
                return anchors
    return anchors


def anchor_relative_dates(text: str, recorded_at: str | None) -> str:
    """Return a compact anchored-date hint for relative time phrases, or '' if none found."""
    anchors = _resolve_relative_anchors(text, recorded_at)
    if not anchors:
        return ""
    notes = [f'"{phrase}"≈{resolved}' for phrase, resolved in anchors[:3]]
    return f" [anchored: {'; '.join(notes)}]"


def extract_anchored_search_tokens(text: str, recorded_at: str | None) -> list[str]:
    """Extract deterministic anchored year/date tokens for FTS5 indexing without mutating raw text."""
    anchors = _resolve_relative_anchors(text, recorded_at)
    tokens: list[str] = []
    for _, resolved in anchors:
        tokens.append(resolved)
        year = resolved[:4]
        if year.isdigit() and year not in tokens:
            tokens.append(year)
    return tokens


def compute_actr_activation(
    access_count: int,
    salience: float = 1.0,
    ref_timestamp: str | None = None,
    now_iso: str | None = None,
) -> float:
    """Cognitive ACT-R Base-Level Activation combining frequency reinforcement & power-law decay."""
    import math

    base_sal = max(0.15, min(2.0, float(salience)))
    freq_term = 0.18 * math.log1p(max(0, int(access_count)))
    age_days = 0.0
    ref_dt = parse_anchor_date(ref_timestamp)
    now_dt = parse_anchor_date(now_iso) if now_iso else datetime.now(UTC).replace(tzinfo=None)
    if ref_dt is not None and now_dt is not None:
        age_days = max(0.0, (now_dt - ref_dt).total_seconds() / 86400.0)
    decay_term = 0.08 * math.log1p(age_days)
    act = base_sal * (1.0 + freq_term - decay_term)
    return max(0.15, min(2.0, round(act, 4)))


class FeatureReranker:
    """Zero-dependency multi-feature + optional API Cross-Encoder reranker with session-burst awareness."""

    def __init__(self) -> None:
        self._api_key = os.getenv("HMPH_RERANK_API_KEY", "").strip()
        self._base_url = os.getenv("HMPH_RERANK_BASE_URL", "").strip()
        self._model = os.getenv("HMPH_RERANK_MODEL", "").strip()

    @property
    def uses_api(self) -> bool:
        return bool(self._api_key and self._base_url and self._model)

    def rerank_messages(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        intent: str,
        limit: int = 12,
        diverse_sessions: bool = False,
    ) -> list[str]:
        """Rerank candidate message dicts and optionally enforce session-burst-aware diversity."""
        if not candidates:
            return []

        if self.uses_api and len(candidates) > 1:
            api_order = self._try_api_rerank(query, candidates, limit=limit)
            if api_order:
                return api_order

        q_tokens = set(tokenize(query))
        q_proper = {
            w.lower()
            for w in re.findall(r"\b[A-Z][a-zA-Z0-9-]{2,}\b|\b\d+\b", query)
        }

        scored: list[tuple[float, dict[str, Any]]] = []
        for cand in candidates:
            sem_r = cand.get("sem_rank")
            fts_r = cand.get("fts_rank")
            ppr = float(cand.get("ppr_score") or 0.0)
            rrf = (1.0 / (25.0 + sem_r) if sem_r else 0.0) + (
                1.0 / (25.0 + fts_r) if fts_r else 0.0
            )
            c_tokens = set(tokenize(cand["content"]))
            lex_cov = (len(q_tokens & c_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
            c_low = cand["content"].lower()
            proper_hit = (
                sum(1 for p in q_proper if p in c_low) / max(1, len(q_proper))
                if q_proper
                else 0.0
            )
            score = rrf * 30.0 + lex_cov * 0.45 + proper_hit * 0.15 + min(ppr * 1.5, 0.15)
            act_boost = float(cand.get("activation") or 0.0)
            if act_boost > 0.0:
                score += min(0.08, (act_boost - 1.0) * 0.08) if act_boost >= 1.0 else max(-0.04, (act_boost - 1.0) * 0.05)
            if intent == "current" and cand.get("is_recent_half"):
                score += 0.05
            scored.append((score, cand))

        scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))

        if not diverse_sessions:
            return [c["id"] for _, c in scored[:limit]]

        # Pass 1 (Session-Burst Aware Diversity):
        # Prefer distinct sessions for wide multi-session coverage, while allowing a 2nd
        # high-confidence event in the same session (score >= 70% of top) so same-day
        # multi-event sessions (e.g., two museum visits on one date) are never crowded out
        # by low-scoring noise sessions.
        selected: list[str] = []
        seen_ids: set[str] = set()
        session_counts: dict[str, int] = {}
        top_score = scored[0][0] if scored else 0.0

        for sc, cand in scored:
            if len(selected) >= limit:
                break
            sid = cand.get("session_id") or ""
            cnt = session_counts.get(sid, 0) if sid else 0
            if cnt == 0:
                if top_score > 0 and sc < top_score * 0.35:
                    continue
            elif cnt == 1:
                if top_score <= 0 or sc < top_score * 0.70:
                    continue
            else:
                continue
            selected.append(cand["id"])
            seen_ids.add(cand["id"])
            if sid:
                session_counts[sid] = cnt + 1

        # Pass 2: Allow moderate secondary events (up to 2 per session) or remaining distinct sessions.
        for sc, cand in scored:
            if len(selected) >= limit:
                break
            if cand["id"] in seen_ids:
                continue
            sid = cand.get("session_id") or ""
            cnt = session_counts.get(sid, 0) if sid else 0
            if sid and cnt >= 2:
                continue
            if sid and cnt == 1 and top_score > 0 and sc < top_score * 0.50:
                continue
            selected.append(cand["id"])
            seen_ids.add(cand["id"])
            if sid:
                session_counts[sid] = cnt + 1

        # Pass 3: Fill any remaining budget strictly by rank.
        for _, cand in scored:
            if len(selected) >= limit:
                break
            if cand["id"] not in seen_ids:
                selected.append(cand["id"])
                seen_ids.add(cand["id"])
        return selected

    def _try_api_rerank(
        self, query: str, candidates: list[dict[str, Any]], *, limit: int
    ) -> list[str] | None:
        try:
            import httpx

            with httpx.Client(timeout=httpx.Timeout(15.0, connect=8.0)) as client:
                resp = client.post(
                    f"{self._base_url.rstrip('/')}/rerank",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": [c["content"][:1500] for c in candidates],
                        "top_n": min(limit, len(candidates)),
                    },
                )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results") or []
            return [
                candidates[int(r["index"])]["id"]
                for r in results
                if isinstance(r, dict) and 0 <= int(r.get("index", -1)) < len(candidates)
            ]
        except Exception:
            return None
