"""Pure, bounded companion decisions. No database, model or inferred profile."""

import re
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from hmph.preference_types import ExpressionPreferences
from hmph.source_policy import retainable
from hmph.working_memory import plan_recall, query_terms


class PersonaCore(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: int = 1
    name: Literal["Hmph"] = "Hmph"
    identity: Literal["fictional_companion"] = "fictional_companion"
    language: Literal["zh"] = "zh"
    style: Literal["mild_tsundere"] = "mild_tsundere"


class MemoryUse(BaseModel):
    source_id: UUID
    permission: Literal["omit", "context_only", "mention"]
    reason: Literal[
        "boundary",
        "sensitive",
        "repeated",
        "requested",
        "continuity",
        "proactive",
        "not_relevant",
        "closing",
        "unclear_repair",
    ]


class CompanionPlan(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: int = 1
    persona_version: int = 1
    policy_revision: int
    act: Literal[
        "acknowledge", "celebrate", "playful", "help", "recall", "clarify", "repair", "close"
    ]
    support_mode: Literal["listen", "suggest", "solve", "unknown"]
    tone: Literal["warm", "gentle", "light"]
    operator: Literal["none", "direct_care", "mild_tease"]
    memory_use: tuple[MemoryUse, ...] = Field(default=(), max_length=160)
    shared_thread_action: Literal["hold", "pause"] = "hold"
    repair_action: Literal["none", "applied", "clarify"] = "none"
    output_limit: int = 256
    max_proactive_memories: int = 1
    signal_source: UUID | None = None
    shared_thread: dict | None = None
    no_active_shared_thread: bool = False
    rhythm: dict = Field(default_factory=dict)
    thread_control: Literal["open", "paused", "resolved", "dropped"] | None = None

    def model_context(self):
        # Keep the release prompt on the reviewed arm without recent-interaction statistics.
        return self.model_copy(update={"rhythm": {}}).model_dump(
            mode="json",
            include={"repair_action", "shared_thread", "no_active_shared_thread", "rhythm",
                     "thread_control"},
        )


@dataclass(frozen=True)
class SourceCandidate:
    id: UUID
    text: str
    recent: bool = False


RECALL = re.compile(
    r"^我(?:现在|目前|最近)?(?:(?:每天|平时|午后|下午|早上|晚上|固定|通常|一般|主要|常常|常|最常|正在|在)){0,3}"
    r"(?:喝|吃|在用|使用|用|养的|养|买的|选的|常用的)"
    r"[^，,。！？?]{0,12}(?:什么|哪个|哪款|哪台|哪只)|"
    r"记得|上次|之前|以前|还记|我.{0,8}(?:叫什|住哪|住在|名字|喜欢|公司)|"
    r"我.{0,30}(?:多少|哪些|哪几|几次|何时|什么时候|哪一?天|几号|哪年|一共|总共)|"
    r"^(?:what|which|who|where|when)\b(?![^?]{0,100}\b(?:should|could|would)\b)"
    r"[^?]{0,80}\b[a-z][a-z-]{1,30}['’]s\b|"
    r"what.*(?:my|\bI\b|\bwe\b)|where.*\bI\b|when did|how many.*\b(?:I|we)\b",
    re.I,
)
CLOSE = re.compile(
    r"^(?:今天|现在)?(?:先这样|不聊了|再见|晚安|我要睡了|bye|good night)"
    r"(?:吧|啦|哦|啊)?(?:[，,。！？!?.\s]|$)",
    re.I,
)
LISTEN = re.compile(
    r"只想.{0,4}(?:倾诉|吐槽|说说)|先听我|别.{0,4}(?:建议|办法)|不要.{0,4}建议|不需要建议"
)
PAUSE = re.compile(
    r"先.{0,16}(?:搁一搁|搁置|暂停|放一放|别聊)|(?:别|不要|不再).{0,6}(?:追问|继续聊)"
)
HELP = re.compile(r"给我.{0,4}建议|帮我.{0,4}(?:分析|解决|想办法)|该怎么办|how (?:do|can) I", re.I)
SENSITIVE = re.compile(r"失恋|创伤|自杀|抑郁|病情|性经历|家暴|银行卡|密码|trauma|diagnosis", re.I)
UNCLEAR = re.compile(r"你.{0,3}(?:又|记|叫|弄|说)错|不是我的|不是我|别再这样")


def requested(query, target=""):
    return bool(RECALL.search(query) or re.search(r"[?？]|告诉我|聊聊|说说", query)) and (
        not target or target.casefold() in query.casefold()
    )


def forbidden_targets(preferences, query):
    return tuple(
        b["target"]
        for b in preferences.get("_boundaries", ())
        if b["mode"] != "on_request" or not requested(query, b["target"])
    )


def nickname_targets(preferences, query):
    active = set(forbidden_targets(preferences, query))
    topics = {b["target"] for b in preferences.get("_boundaries", ()) if b["kind"] == "topic"}
    return {
        b["target"]
        for b in preferences.get("_boundaries", ())
        if b["kind"] == "nickname" and b["target"] in active
    } - topics


def has_forbidden(text, targets, *, nickname=None, name_targets=()):
    # An explicitly selected longer nickname may contain the former nickname.
    # Only nickname rules get this exception; topic boundaries still win.
    without_name = text.replace(nickname, " ") if nickname else text
    return any(
        t.casefold() in (without_name if t in name_targets else text).casefold() for t in targets
    )


def forbidden_in(preferences, query, text):
    return has_forbidden(
        text,
        forbidden_targets(preferences, query),
        nickname=preferences.get("nickname"),
        name_targets=nickname_targets(preferences, query),
    )


def safe_excerpt(text, targets, *, nickname=None, name_targets=()):
    """Only whole contiguous clauses. Never synthesize an alleged user quotation."""
    if not retainable(text):
        return None
    if not has_forbidden(text, targets, nickname=nickname, name_targets=name_targets):
        return 0, len(text)
    spans = [(m.start(), m.end()) for m in re.finditer(r"[^，,。；;！？!?\n]+", text)]
    spans = [
        (a, b)
        for a, b in spans
        if text[a:b].strip()
        and not has_forbidden(text[a:b], targets, nickname=nickname, name_targets=name_targets)
    ]
    return max(spans, key=lambda s: s[1] - s[0], default=None)


def decide(
    query,
    preferences,
    candidates,
    *,
    policy_revision,
    recent_mentions=(),
    repaired=False,
    unclear_repair=False,
    mode_hint=None,
    signal_source=None,
    shared_source_id=None,
    contextual_recall=False,
):
    prefs = ExpressionPreferences.model_validate(
        {k: v for k, v in preferences.items() if k in ExpressionPreferences.model_fields}
    )
    close, listen = bool(CLOSE.search(query.strip())), bool(LISTEN.search(query))
    pause = bool(PAUSE.search(query))
    unclear = not repaired and (unclear_repair or bool(UNCLEAR.search(query)))
    # A question about a retrieved event can omit "I/my" entirely. Use the
    # existing query plan and content words, while keeping advice/hypotheses out.
    terms = set(query_terms(query))
    grounded_question = (
        plan_recall(query).intent != "social"
        and bool(
            re.search(
                r"\b(?:what|which|who|where|when)\b|什么|哪个|哪一|哪天|何时|多少|[?？]",
                query,
                re.I,
            )
        )
        and not re.search(r"\b(?:should|could|would|if)\b|如果|假如|应该|建议", query, re.I)
        and any(len(terms.intersection(query_terms(c.text))) >= 2 for c in candidates)
    )
    recall = (bool(RECALL.search(query)) or grounded_question or contextual_recall) and not (
        close or unclear
    )
    help_requested = bool(HELP.search(query)) and not listen
    celebrate = bool(
        re.search(r"好消息|拿到.{0,6}offer|通过.{0,6}考试|升职了|被录取了|庆祝", query, re.I)
    )
    if close:
        act = "close"
    elif repaired:
        act = "repair"
    elif unclear:
        act = "clarify"
    elif listen:
        act = "acknowledge"
    elif help_requested:
        act = "help"
    elif celebrate:
        act = "celebrate"
    else:
        act = "recall" if recall else "acknowledge"
    mode = (
        "listen"
        if listen
        else "solve"
        if help_requested
        else (
            "listen"
            if prefs.support_mode == "listen"
            else "suggest"
            if prefs.support_mode == "suggest"
            else "unknown"
        )
    )
    explicit_support = "support_mode" in preferences.get(
        "_field_sources", {"support_mode": True} if preferences.get("_source_memory_id") else {}
    )
    used_signal = (
        not listen
        and not help_requested
        and not explicit_support
        and mode_hint in {"listen", "advice", "solve"}
    )
    if used_signal:
        mode = {"listen": "listen", "advice": "suggest", "solve": "solve"}[mode_hint]
    targets, terms = forbidden_targets(preferences, query), set(query_terms(query))
    decisions, proactive = [], False
    for source in candidates:
        span = safe_excerpt(
            source.text,
            targets,
            nickname=preferences.get("nickname"),
            name_targets=nickname_targets(preferences, query),
        )
        visible = source.text[slice(*span)] if span else ""
        if span is None:
            permission, reason = "omit", "boundary"
        elif close or unclear or pause:
            permission, reason = "omit", "closing" if close or pause else "unclear_repair"
        elif SENSITIVE.search(visible) and not (
            recall and terms.intersection(query_terms(visible))
        ):
            permission, reason = "omit", "sensitive"
        elif source.id in recent_mentions and not recall:
            permission, reason = "omit", "repeated"
        elif recall:
            permission, reason = "mention", "requested"
        elif (
            not proactive
            and not listen
            and (terms.intersection(query_terms(visible)) or source.id == shared_source_id)
        ):
            permission, reason, proactive = "mention", "proactive", True
        elif source.recent:
            permission, reason = "context_only", "continuity"
        else:
            permission, reason = "omit", "not_relevant"
        decisions.append(MemoryUse(source_id=source.id, permission=permission, reason=reason))
    gentle = mode == "listen" or repaired or unclear or bool(SENSITIVE.search(query))
    return CompanionPlan(
        policy_revision=policy_revision,
        act=act,
        support_mode=mode,
        tone="gentle" if gentle else "warm",
        operator="direct_care" if gentle else "none",
        memory_use=tuple(decisions),
        shared_thread_action="pause" if close or pause else "hold",
        repair_action="applied" if repaired else "clarify" if unclear else "none",
        signal_source=signal_source if used_signal else None,
    )
