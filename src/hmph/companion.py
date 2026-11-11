"""Source-backed controls and publication bookkeeping around the pure policy."""

import re
from types import SimpleNamespace
from uuid import uuid5

from sqlalchemy import select

from hmph.companion_policy import CompanionPlan, decide
from hmph.dialogue import state_sources
from hmph.models import (
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


async def apply_controls(session, pair, turn, message, *, proposed=None):
    patch, boundaries = parse_controls(message.content) if proposed is None else proposed
    unclear = bool(
        re.fullmatch(r"你又?(?:记|叫|弄|说)错了[。！!]?|别再这样[。！!]?", message.content)
    )
    unclear |= any(re.match(r"那|这|某|他|她|不喜欢的人", b["target"]) for b in boundaries)
    if unclear:
        patch, boundaries = {}, []
    if not patch and not boundaries and not unclear:
        return False
    previous = dict(pair.preferences)
    revoked_names = {item["target"] for item in boundaries if item["kind"] == "nickname"}
    if "nickname" not in patch and previous.get("nickname") in revoked_names:
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
    turn.companion_plan = {
        **(turn.companion_plan or {}),
        "controls_applied": not unclear,
        "controls_unclear": unclear,
    }
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
    cue = short_term.data.response_mode
    if cue:
        source = originals.get(cue.evidence.message_id)
        if source is None or cue.evidence.quote not in source.content:
            cue = None
    plan = decide(
        query,
        pair.preferences,
        (),
        policy_revision=pair.policy_revision,
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
    if shared_thread:
        plan = plan.model_copy(
            update={
                "shared_thread": {
                    "id": str(shared_thread.id),
                    "topic": shared_thread.topic,
                    "source_id": str(shared_thread.source_message_id),
                    "revision": shared_thread.revision,
                    "status": shared_thread.status,
                }
            }
        )
    safe_rows = [
        SimpleNamespace(
            id=row.id,
            revision=row.revision,
            source_seq=row.source_seq,
            role=row.role,
            content=row.content,
            turn_id=row.turn_id,
            start_offset=0,
            end_offset=len(row.content),
        )
        for row in rows
        if row.id in originals
    ]
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
        if hit.sources and all(
            s.message_id in originals and s.quote in originals[s.message_id].content
            for s in hit.sources
        ):
            safe_memories.append(hit)
    safe_raw = []
    for source in candidates.raw:
        original = originals.get(source.message_id)
        if original is None or original.revision != source.message_revision:
            continue
        if not 0 <= source.start_offset < source.end_offset <= len(original.content):
            continue
        if source.content == original.content[source.start_offset : source.end_offset]:
            safe_raw.append(source)
    data = short_term.data.model_copy(deep=True)

    def allowed(anchor):
        return (
            anchor is not None
            and anchor.message_id in originals
            and anchor.quote in originals[anchor.message_id].content
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


async def eligible_mentions(session, turn, body, mentions):
    result, _ = await checked_mentions(session, turn, body, mentions)
    return result


async def checked_mentions(session, turn, body, mentions):
    from hmph.models import TurnSourceUse

    result = []
    current_mentions = []
    missing_quote = set()
    for hint in mentions:
        if hint.quote not in body:
            continue
        use = await session.get(TurnSourceUse, (turn.id, hint.source_id))
        source = await session.get(Message, hint.source_id)
        scope = (turn.memory_context or {}).get("current_source", {})
        start, end = scope.get("start_offset"), scope.get("end_offset")
        if (
            source is not None
            and source.turn_id == turn.id
            and source.role == "user"
            and not source.deleted
            and use is not None
            and str(source.id) == scope.get("message_id")
            and source.revision == use.message_revision == scope.get("revision")
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(source.content)
            and (hint.source_quote or hint.quote) in source.content[start:end]
        ):
            # This is acknowledgment of the actual current user input, not a
            # proactive old-memory use. Keep its publication/deletion lineage.
            current_mentions.append((source.id, source.revision))
            continue
        if use is None:
            entry = await session.get(MemoryEntry, hint.source_id)
            if entry is not None and entry.pair_id == turn.pair_id:
                return None, "memory_id_as_source"
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
    return list(dict.fromkeys([*current_mentions, *result])), "ok"
