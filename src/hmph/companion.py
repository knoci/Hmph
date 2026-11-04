"""Source-backed controls and publication bookkeeping around the pure policy."""

import re
from types import SimpleNamespace
from uuid import uuid5

from sqlalchemy import select

from hmph.companion_policy import (
    CompanionPlan,
    SourceCandidate,
    decide,
    forbidden_in,
    forbidden_targets,
    nickname_targets,
    safe_excerpt,
)
from hmph.dialogue import state_sources
from hmph.models import (
    CompanionMention,
    MemoryEntry,
    MemorySource,
    MemoryTask,
    MemoryVersion,
    Message,
    RepairState,
    Turn,
)
from hmph.preference_types import ExpressionPreferences
from hmph.source_policy import retainable


def field_sources(raw):
    return raw.get(
        "_field_sources",
        {field: raw["_source_memory_id"] for field in ExpressionPreferences.model_fields}
        if raw.get("_source_memory_id")
        else {},
    )


def parse_controls(text):
    """Narrow explicit imperatives only; quotes/hypotheses/third parties don't mutate state."""
    if (
        not retainable(text)
        or len(text) > 400
        or re.search(r"如果|假如|例如|他说|她说|朋友|[\"“”‘’]|\bif\b", text, re.I)
    ):
        return {}, []
    patch, boundaries = {}, []
    clauses = [s.strip() for s in re.split(r"[，,。；;！!\n]", text) if s.strip()]
    for clause in clauses:
        if m := re.fullmatch(r"(?:以后|今后|从现在起)(?:请)?叫我([^\s?？]{1,20})", clause):
            patch["nickname"] = m[1]
        elif re.fullmatch(
            r"(?:以后|今后)(?:请)?(?:先听我说|不要(?:主动)?给我建议|别给我建议)", clause
        ):
            patch["support_mode"] = "listen"
        elif re.fullmatch(r"(?:以后|今后)(?:请)?(?:少开玩笑|不要开玩笑)", clause):
            patch["playfulness"] = "low"
        elif re.fullmatch(r"(?:以后|今后)(?:请)?(?:回复简短(?:一点|一些|点)?|少说点)", clause):
            patch["reply_length"] = "short"
        elif m := re.fullmatch(
            r"(?:以后|今后)?(?:请)?(?:不要再|别再)叫我(.{1,40}?)(?:了)?", clause
        ):
            boundaries.append({"target": m[1], "kind": "nickname", "mode": "never"})
        elif m := re.fullmatch(
            r"(?:以后|今后)?(?:请)?(?:不要|别)(主动)?提(?:起)?(?:我的)?(.{2,40})", clause
        ):
            boundaries.append(
                {"target": m[2], "kind": "topic", "mode": "on_request" if m[1] else "never"}
            )
        else:
            return {}, []
    return patch, boundaries


async def apply_controls(session, pair, turn, message):
    patch, boundaries = parse_controls(message.content)
    unclear = bool(
        re.fullmatch(r"你又?(?:记|叫|弄|说)错了[。！!]?|别再这样[。！!]?", message.content)
    )
    unclear |= any(re.match(r"那|这|某|他|她|不喜欢的人", b["target"]) for b in boundaries)
    if unclear:
        patch, boundaries = {}, []
    if not patch and not boundaries and not unclear:
        return False
    previous = dict(pair.preferences)
    if "nickname" not in patch and previous.get("nickname") in {b["target"] for b in boundaries}:
        patch["nickname"] = None
    if (
        patch.get("nickname")
        and previous.get("nickname")
        and patch["nickname"] != previous["nickname"]
    ):
        boundaries.append({"target": previous["nickname"], "kind": "nickname", "mode": "never"})
    if patch.get("nickname"):
        old_names = list(
            await session.scalars(
                select(MemoryVersion.value)
                .join(MemoryEntry)
                .where(
                    MemoryEntry.pair_id == pair.id,
                    MemoryEntry.predicate == "name",
                    MemoryEntry.status == "active",
                    MemoryEntry.scope == "current",
                    MemoryVersion.revision == MemoryEntry.revision,
                )
                .limit(17)
            )
        )
        boundaries += [
            {"target": value, "kind": "nickname", "mode": "never"}
            for value in old_names
            if value != patch["nickname"] and len(value) <= 40
        ]
    existing = previous.get("_boundaries", [])
    indexed = {(b["kind"], b["target"]): b for b in existing}
    if patch.get("nickname"):
        indexed.pop(("nickname", patch["nickname"]), None)
    if len(set(indexed) | {(b["kind"], b["target"]) for b in boundaries}) > 16:
        from hmph.contracts import DomainError

        raise DomainError("companion_control_capacity", 422)
    memory_id = uuid5(message.id, "companion.controls.v1")
    task = await session.scalar(
        select(MemoryTask).where(MemoryTask.source_message_id == message.id)
    )
    entry = MemoryEntry(
        id=memory_id,
        pair_id=pair.id,
        kind="fact",
        predicate="companion_repair"
        if unclear
        else "expression_preferences"
        if patch
        else "companion_boundary",
        scope="current",
        source_order=task.source_order,
        revision=1,
    )
    session.add(entry)
    await session.flush()
    session.add(MemoryVersion(memory_id=entry.id, revision=1, value=message.content))
    await session.flush()
    session.add(
        MemorySource(
            memory_id=entry.id,
            revision=1,
            message_id=message.id,
            message_revision=message.revision,
            quote=message.content,
        )
    )
    sources = {**field_sources(previous), **dict.fromkeys(patch, str(memory_id))}
    indexed.update(
        {(b["kind"], b["target"]): {**b, "source_memory_id": str(memory_id)} for b in boundaries}
    )
    if not unclear:
        pair.preferences = {
            **previous,
            **patch,
            "_field_sources": sources,
            "_source_memory_id": str(memory_id),
            "_boundaries": list(indexed.values()),
        }
        pair.policy_revision += 1
    turn.policy_revision = pair.policy_revision
    kinds = ("unclear",) if unclear else (*patch, *(f"boundary_{b['kind']}" for b in boundaries))
    for kind in dict.fromkeys(kinds):
        session.add(
            RepairState(
                source_message_id=message.id,
                kind=kind,
                pair_id=pair.id,
                source_memory_id=entry.id,
                source_revision=message.revision,
                policy_revision=pair.policy_revision,
                status="open" if unclear else "corrected",
            )
        )
    turn.companion_plan = {"controls_applied": not unclear, "controls_unclear": unclear}
    return not unclear


def repaired(turn):
    return bool(turn.companion_plan and turn.companion_plan.get("controls_applied"))


def plan_view(turn):
    if turn.companion_plan and "act" in turn.companion_plan:
        return CompanionPlan.model_validate(
            {k: v for k, v in turn.companion_plan.items() if k in CompanionPlan.model_fields}
        )
    return None


def forget_controls(pair, memories):
    removed = {str(mid) for mid in memories}
    before = dict(pair.preferences)
    sources = dict(field_sources(before))
    raw = dict(before)
    for field, mid in list(sources.items()):
        if mid in removed:
            raw.pop(field, None)
            sources.pop(field)
    boundaries = [b for b in before.get("_boundaries", []) if b["source_memory_id"] not in removed]
    if sources != field_sources(before) or boundaries != before.get("_boundaries", []):
        raw["_field_sources"], raw["_boundaries"] = sources, boundaries
        if raw.get("_source_memory_id") in removed:
            raw["_source_memory_id"] = next(iter(sources.values()), None) or (
                boundaries[-1]["source_memory_id"] if boundaries else None
            )
        pair.preferences = raw
        pair.policy_revision += 1


async def project_context(
    session, pair, turn, query, rows, candidates, short_term, *, shared_thread=None
):
    """One projection for every prompt channel, including original message bodies."""
    current = rows[0].id
    ids = {row.id for row in rows}
    ids |= {s.message_id for m in candidates.memories for s in m.sources}
    ids |= {s.message_id for s in candidates.raw}
    ids |= set(state_sources(short_term.data))
    messages = list(
        await session.scalars(
            select(Message)
            .join(Turn)
            .where(Message.id.in_(ids), Turn.pair_id == pair.id, Message.deleted.is_(False))
            .order_by((Message.id != current).asc(), Message.id)
            .limit(160)
        )
    )
    originals = {m.id: m for m in messages}
    priority = {
        s.message_id: i for i, s in enumerate(s for hit in candidates.memories for s in hit.sources)
    }
    messages.sort(
        key=lambda m: (
            -1
            if shared_thread and m.id == shared_thread.source_message_id
            else priority.get(m.id, 1000),
            str(m.id),
        )
    )
    control_ids = set(
        await session.scalars(
            select(MemorySource.message_id)
            .join(MemoryEntry, MemorySource.memory_id == MemoryEntry.id)
            .where(
                MemorySource.message_id.in_(ids),
                MemoryEntry.pair_id == pair.id,
                MemoryEntry.predicate.in_(
                    ("expression_preferences", "companion_boundary", "companion_repair")
                ),
            )
        )
    )
    from hmph.models import SharedThread, SharedThreadSource

    control_ids |= set(
        await session.scalars(
            select(SharedThreadSource.message_id)
            .join(SharedThread)
            .where(
                SharedThread.pair_id == pair.id,
                (SharedThread.status != "open")
                | (SharedThreadSource.message_id != SharedThread.source_message_id),
            )
        )
    )
    previous_turn = await session.scalar(
        select(Turn.id)
        .where(
            Turn.pair_id == pair.id,
            Turn.id != turn.id,
            Turn.published_segments > 0,
            Turn.created_at < turn.created_at,
        )
        .order_by(Turn.created_at.desc())
        .limit(1)
    )
    mentioned = set(
        await session.scalars(
            select(CompanionMention.source_message_id).where(
                CompanionMention.turn_id == previous_turn
            )
        )
    )
    if shared_thread and shared_thread.source_message_id in mentioned:
        # A later explicit resume is fresh user intent; once actually mentioned
        # again, the ordinary previous-publication suppression applies again.
        latest_control = await session.scalar(
            select(Turn.created_at)
            .join(Message)
            .join(SharedThreadSource, SharedThreadSource.message_id == Message.id)
            .where(SharedThreadSource.thread_id == shared_thread.id)
            .order_by(Turn.created_at.desc())
            .limit(1)
        )
        prior = await session.get(Turn, previous_turn)
        if latest_control is not None and latest_control > prior.created_at:
            mentioned.discard(shared_thread.source_message_id)
    recent = {row.id for row in rows}
    cue = short_term.data.response_mode
    if cue:
        original = originals.get(cue.evidence.message_id)
        span = (
            safe_excerpt(
                original.content,
                forbidden_targets(pair.preferences, query),
                nickname=pair.preferences.get("nickname"),
                name_targets=nickname_targets(pair.preferences, query),
            )
            if original
            else None
        )
        if (
            span is None
            or cue.evidence.message_id in control_ids
            or cue.evidence.quote not in original.content[slice(*span)]
        ):
            cue = None
    plan = decide(
        query,
        pair.preferences,
        [
            SourceCandidate(
                m.id, m.content, m.id in recent or m.id in candidates.query_hint_source_ids
            )
            for m in messages
            if m.id != current and m.role == "user" and m.id not in control_ids
        ],
        policy_revision=pair.policy_revision,
        recent_mentions=mentioned,
        repaired=repaired(turn),
        unclear_repair=bool(turn.companion_plan and turn.companion_plan.get("controls_unclear")),
        mode_hint=cue.mode if cue else None,
        signal_source=cue.evidence.message_id if cue else None,
        shared_source_id=shared_thread.source_message_id if shared_thread else None,
        contextual_recall=bool(candidates.query_hint_source_ids),
    )
    from hmph.shared_threads import interaction_rhythm

    previous_messages = list(
        await session.scalars(
            select(Message)
            .join(Turn)
            .where(
                Turn.pair_id == pair.id,
                Turn.id != turn.id,
                Turn.published_segments > 0,
                Message.role == "assistant",
                Message.deleted.is_(False),
                Turn.created_at < turn.created_at,
            )
            .order_by(Turn.created_at.desc())
            .limit(2)
        )
    )
    plan = plan.model_copy(
        update={"rhythm": interaction_rhythm([m.content for m in previous_messages])}
    )
    thread_control = (turn.companion_plan or {}).get("thread_control")
    if thread_control:
        plan = plan.model_copy(
            update={
                "thread_control": thread_control,
                "shared_thread_action": "pause"
                if thread_control == "paused"
                else plan.shared_thread_action,
            }
        )
    from sqlalchemy.dialects.postgresql import insert

    from hmph.models import TurnSourceUse

    for previous in previous_messages:
        await session.execute(
            insert(TurnSourceUse)
            .values(turn_id=turn.id, message_id=previous.id, message_revision=previous.revision)
            .on_conflict_do_nothing()
        )
    if shared_thread and any(
        d.source_id == shared_thread.source_message_id and d.permission == "mention"
        for d in plan.memory_use
    ):
        plan = plan.model_copy(
            update={
                "shared_thread": {
                    "id": str(shared_thread.id),
                    "topic": shared_thread.topic,
                    "source_id": str(shared_thread.source_message_id),
                    "revision": shared_thread.revision,
                    "instruction": "合适时自然承接这一项共同计划；不要编造进展或催促。",
                }
            }
        )
    permissions = {d.source_id: d.permission for d in plan.memory_use}
    targets = forbidden_targets(pair.preferences, query)
    spans = {}
    for mid, message in originals.items():
        if mid in control_ids or permissions.get(mid) == "omit":
            continue
        span = safe_excerpt(
            message.content,
            targets,
            nickname=pair.preferences.get("nickname"),
            name_targets=nickname_targets(pair.preferences, query),
        )
        if span is not None:
            spans[mid] = span
    safe_rows = []
    for row in rows:
        span = spans.get(row.id)
        if span is None:
            if row.id != current:
                continue
            content = "[用户表达了互动边界或修正意图，请依据表达计划回应，不复述受限内容。]"
        else:
            content = row.content[span[0] : span[1]]
        # Never mutate ORM Message.content: the original is the user's source.
        safe_rows.append(
            SimpleNamespace(
                id=row.id,
                revision=row.revision,
                source_seq=row.source_seq,
                role=row.role,
                content=content,
                turn_id=row.turn_id,
                start_offset=span[0] if span else None,
                end_offset=span[1] if span else None,
            )
        )
    from hmph.models import TurnSourceUse

    omitted = {d.source_id for d in plan.memory_use if d.permission == "omit"} | control_ids
    assistant_turns = {r.turn_id for r in safe_rows if r.role == "assistant"}
    if omitted and assistant_turns:
        blocked = set(
            await session.scalars(
                select(TurnSourceUse.turn_id).where(
                    TurnSourceUse.turn_id.in_(assistant_turns),
                    TurnSourceUse.message_id.in_(omitted),
                )
            )
        )
        safe_rows = [r for r in safe_rows if r.role != "assistant" or r.turn_id not in blocked]
    # A boundary can affect an assistant paraphrase whose words differ. Legacy
    # replies without sufficient provenance are conservatively omitted too.
    if targets:
        safe_rows = [r for r in safe_rows if r.role == "user"]
    safe_memories = []
    for hit in candidates.memories:
        if hit.predicate in {
            "name",
            "nickname",
            "expression_preferences",
            "companion_boundary",
            "companion_repair",
        }:
            if hit.predicate in {
                "expression_preferences",
                "companion_boundary",
                "companion_repair",
            }:
                continue
            if pair.preferences.get("nickname") and hit.value != pair.preferences["nickname"]:
                continue
        if forbidden_in(pair.preferences, query, hit.value):
            continue
        if hit.sources and all(
            s.message_id in spans
            and s.quote in originals[s.message_id].content[slice(*spans[s.message_id])]
            for s in hit.sources
        ):
            safe_memories.append(hit)
    safe_raw = []
    for source in candidates.raw:
        if source.message_id not in spans:
            continue
        # Intersect the permitted original span with the exact retrieved excerpt.
        start, end = spans[source.message_id]
        start, end = max(start, source.start_offset), min(end, source.end_offset)
        if start < end:
            body = originals[source.message_id].content
            safe_raw.append(
                source.model_copy(
                    update={
                        "content": body[start:end],
                        "start_offset": start,
                        "end_offset": end,
                        "truncated": start > 0 or end < len(body),
                    }
                )
            )
    data = short_term.data.model_copy(deep=True)

    def allowed(anchor):
        return (
            anchor is not None
            and anchor.message_id in spans
            and anchor.quote
            in originals[anchor.message_id].content[slice(*spans[anchor.message_id])]
        )

    data.topic = data.topic if allowed(data.topic) else None
    data.goal = data.goal if allowed(data.goal) else None
    data.entities = [a for a in data.entities if allowed(a)]
    data.open_items = [
        item
        for item in data.open_items
        if allowed(item.content)
        and (item.transition_evidence is None or allowed(item.transition_evidence))
    ]
    data.response_mode = (
        data.response_mode if data.response_mode and allowed(data.response_mode.evidence) else None
    )
    candidates.memories, candidates.raw = safe_memories, safe_raw
    return safe_rows, candidates, short_term.model_copy(update={"data": data}), plan


def output_block_reason(pair, turn, query, body):
    plan = plan_view(turn)
    if forbidden_in(pair.preferences, query, body):
        return "boundary"
    if plan and plan.act == "close" and re.search(r"[?？]", body):
        return "closing_question"
    if (
        plan
        and plan.support_mode == "listen"
        and re.search(r"建议你|你应该|你可以试|不妨|you should", body, re.I)
    ):
        return "listening_advice"
    return None


def output_allowed(pair, turn, query, body):
    return output_block_reason(pair, turn, query, body) is None


async def eligible_mentions(session, turn, body, mentions):
    result, _ = await checked_mentions(session, turn, body, mentions)
    return result


async def checked_mentions(session, turn, body, mentions):
    from hmph.models import TurnSourceUse

    plan = plan_view(turn)
    allowed = {d.source_id for d in plan.memory_use if d.permission == "mention"} if plan else set()
    result = []
    missing_quote = set()
    for hint in mentions:
        if hint.quote not in body:
            continue
        use = await session.get(TurnSourceUse, (turn.id, hint.source_id))
        source = await session.get(Message, hint.source_id)
        if hint.source_id not in allowed:
            entry = await session.get(MemoryEntry, hint.source_id)
            if entry is not None and entry.pair_id == turn.pair_id:
                return None, "memory_id_as_source"
            return None, "source_permission"
        if use is None:
            return None, "source_dependency"
        if source is None or source.deleted or source.role != "user":
            return None, "source_invalid"
        if source.revision != use.message_revision:
            return None, "source_revision"
        if (hint.source_quote or hint.quote) not in source.content:
            missing_quote.add(source.id)
            continue
        result.append((source.id, source.revision))
    result = list(dict.fromkeys(result))
    if missing_quote - {source_id for source_id, _ in result}:
        return None, "source_quote"
    if (
        plan
        and len(result) > 1
        and not all(
            d.reason == "requested"
            for d in plan.memory_use
            if d.source_id in {r[0] for r in result}
        )
    ):
        return None, "proactive_limit"
    return result, "ok"
