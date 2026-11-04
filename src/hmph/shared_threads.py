"""Explicit source-backed shared plans, reusing existing transactions and forgetting."""

import re

from sqlalchemy import func, select

from hmph.companion_policy import CLOSE, LISTEN, SENSITIVE, forbidden_in
from hmph.contracts import ACTIVE, DomainError
from hmph.dialogue import queue_dialogue
from hmph.models import MemoryTask, Message, Pair, SharedThread, SharedThreadSource, Turn
from hmph.persistence import append_event, terminate_turn
from hmph.source_policy import retainable
from hmph.working_memory import RawEvidence, query_terms

RETURNING = re.compile(r"我回来了|又见面|早上好|晚上好|^你好|^hello\b|^hi\b", re.I)
TRANSITIONS = {
    "open": {"paused", "resolved", "dropped"},
    "paused": {"open", "resolved", "dropped"},
    "resolved": set(),
    "dropped": set(),
}


def interaction_rhythm(texts):
    """A bounded view of actual publication, never planned or cancelled drafts."""
    return {
        "recent_questions": sum(bool(re.search(r"[?？]", text)) for text in texts),
        "question_limit": 0
        if len(texts) >= 2 and all(re.search(r"[?？]", t) for t in texts)
        else 1,
        "avoid_repeated_advice": any(re.search(r"建议|可以试试|不妨", t) for t in texts),
        "instruction": "先承接本轮表达；避免连续追问和重复建议，用户结束时自然收尾。",
    }


class SharedThreads:
    def __init__(self, conversation):
        self.conversation = conversation
        self.db, self.ledger = conversation.db, conversation.ledger

    @staticmethod
    def view(thread):
        return {
            "id": str(thread.id),
            "topic": thread.topic,
            "status": thread.status,
            "revision": thread.revision,
            "source_message_id": str(thread.source_message_id),
        }

    async def read(self, owner_id, character_id):
        async with self.db.sessions() as session:
            rows = await session.scalars(
                select(SharedThread)
                .join(Pair)
                .where(Pair.owner_id == owner_id, Pair.character_id == character_id)
                .order_by(SharedThread.id)
                .limit(32)
            )
            return {"items": [self.view(t) for t in rows]}

    async def sources(self, owner_id, thread_id):
        async with self.db.sessions() as session:
            thread = await session.scalar(
                select(SharedThread)
                .join(Pair)
                .where(SharedThread.id == thread_id, Pair.owner_id == owner_id)
            )
            if thread is None:
                raise DomainError("not_found", 404)
            rows = await session.execute(
                select(Message, Turn.created_at)
                .join(Turn)
                .join(SharedThreadSource, SharedThreadSource.message_id == Message.id)
                .where(
                    SharedThreadSource.thread_id == thread.id,
                    Turn.pair_id == thread.pair_id,
                    Message.deleted.is_(False),
                    Message.role == "user",
                )
                .order_by(Turn.created_at, Message.id)
                .limit(32)
            )
            return {
                "items": [
                    {
                        "source_id": str(m.id),
                        "revision": m.revision,
                        "recorded_at": at.isoformat(),
                        "content": m.content,
                    }
                    for m, at in rows
                ]
            }

    async def write(
        self,
        owner_id,
        chat_id,
        request_id,
        *,
        topic=None,
        thread_id=None,
        expected_revision=None,
        status="open",
        model_mode="fake",
    ):
        if topic is not None and (
            not topic.strip() or len(topic) > 120 or "\x00" in topic or not retainable(topic)
        ):
            raise DomainError("invalid_thread_topic", 422)
        if not thread_id and topic is None:
            raise DomainError("invalid_thread_topic", 422)
        digest = self.ledger.digest(
            str(chat_id),
            {
                "topic": topic,
                "thread_id": str(thread_id) if thread_id else None,
                "expected_revision": expected_revision,
                "status": status,
            },
        )
        async with self.db.sessions.begin() as session:
            await self.ledger.owner_lock(session, owner_id)
            chat = await self.conversation._chat(session, owner_id, chat_id)
            pair = await session.scalar(
                select(Pair).where(Pair.id == chat.pair_id).with_for_update()
            )
            existing = await self.ledger.request(
                session, owner_id, "shared-thread.write", request_id, digest
            )
            if existing:
                old = await session.get(SharedThread, existing)
                if old is None:
                    raise DomainError("not_found", 404)
                return self.view(old)
            thread = await session.get(SharedThread, thread_id) if thread_id else None
            if thread_id and (thread is None or thread.pair_id != pair.id):
                raise DomainError("not_found", 404)
            if thread is not None:
                if thread.revision != expected_revision:
                    raise DomainError("thread_revision_conflict")
                if status not in TRANSITIONS[thread.status]:
                    raise DomainError("thread_transition_conflict")
                count = await session.scalar(
                    select(func.count())
                    .select_from(SharedThreadSource)
                    .where(SharedThreadSource.thread_id == thread.id)
                )
                if count >= 32:
                    raise DomainError("thread_source_limit", 422)
            else:
                count = await session.scalar(
                    select(func.count())
                    .select_from(SharedThread)
                    .where(SharedThread.pair_id == pair.id)
                )
                if count >= 32:
                    raise DomainError("thread_limit", 422)
                if await session.scalar(
                    select(SharedThread.id).where(
                        SharedThread.pair_id == pair.id, SharedThread.topic == topic.strip()
                    )
                ):
                    raise DomainError("thread_exists")
            if status == "open":
                active = await session.scalar(
                    select(func.count())
                    .select_from(SharedThread)
                    .where(SharedThread.pair_id == pair.id, SharedThread.status == "open")
                )
                if active >= 3:
                    raise DomainError("active_thread_limit", 422)
            pair.policy_revision += 1
            for active in list(
                await session.scalars(
                    select(Turn)
                    .where(Turn.pair_id == pair.id, Turn.status.in_(ACTIVE))
                    .with_for_update()
                )
            ):
                await terminate_turn(session, active, "invalidated", "shared_thread_changed")
            now = await session.scalar(select(func.clock_timestamp()))
            control = Turn(
                conversation_id=chat.id,
                pair_id=pair.id,
                source_seq=chat.next_source_seq,
                policy_revision=pair.policy_revision,
                memory_epoch=pair.memory_epoch,
                source_timezone=self.conversation.source_timezone,
                deadline=now,
                created_at=now,
                status="completed",
                model_mode=model_mode,
            )
            chat.next_source_seq += 2
            session.add(control)
            await session.flush()
            label = {"open": "继续", "paused": "暂停", "resolved": "完成", "dropped": "放下"}[
                status
            ]
            body = (
                f"用户明确设定共同计划：{topic.strip()}。"
                if thread is None
                else f"用户明确选择{label}共同计划：{thread.topic}。"
            )
            message = Message(
                conversation_id=chat.id,
                turn_id=control.id,
                source_seq=control.source_seq,
                role="user",
                content=body,
            )
            session.add(message)
            await session.flush()
            if thread is None:
                thread = SharedThread(
                    pair_id=pair.id,
                    topic=topic.strip(),
                    status="open",
                    source_message_id=message.id,
                )
                session.add(thread)
                await session.flush()
            else:
                thread.status, thread.revision = status, thread.revision + 1
            session.add(SharedThreadSource(thread_id=thread.id, message_id=message.id))
            session.add(
                MemoryTask(
                    pair_id=pair.id,
                    source_message_id=message.id,
                    source_revision=1,
                    source_order=pair.next_memory_seq,
                    memory_epoch=pair.memory_epoch,
                    model_mode=model_mode,
                    status="done",
                )
            )
            pair.next_memory_seq += 1
            append_event(session, control, "turn.completed", {"operation": "shared_thread.updated"})
            await queue_dialogue(session, control, control.source_seq + 1)
            self.ledger.save(
                session, owner_id, "shared-thread.write", request_id, digest, thread.id
            )
            return self.view(thread)


async def prepare_thread(session, pair, turn, query, candidates):
    if CLOSE.search(query.strip()) or LISTEN.search(query) or SENSITIVE.search(query):
        return None
    threads = list(
        await session.scalars(
            select(SharedThread)
            .where(SharedThread.pair_id == pair.id, SharedThread.status == "open")
            .order_by(SharedThread.id)
            .limit(3)
        )
    )
    for thread in threads:
        if forbidden_in(pair.preferences, query, thread.topic):
            continue
        if not (
            RETURNING.search(query) or set(query_terms(query)) & set(query_terms(thread.topic))
        ):
            continue
        source = await session.get(Message, thread.source_message_id)
        if source is None or source.deleted:
            continue
        if source.turn_id == turn.id:
            # The user's new command is already the current input, not a past
            # experience to proactively mention or register a second time.
            continue
        start = source.content.find(thread.topic)
        if start < 0:
            continue
        if not any(s.message_id == source.id for s in candidates.raw):
            if candidates.candidate_count >= 80:
                return None
            candidates.candidate_count += 1
            origin = await session.get(Turn, source.turn_id)
            candidates.raw.insert(
                0,
                RawEvidence(
                    message_id=source.id,
                    message_revision=source.revision,
                    role="user",
                    recorded_at=origin.created_at,
                    content=thread.topic,
                    start_offset=start,
                    end_offset=start + len(thread.topic),
                    truncated=True,
                    source_timezone=origin.source_timezone,
                ),
            )
        return thread
    return None


async def apply_thread_command(session, pair, turn, message):
    """Only complete, explicit commands. No quote, hypothesis or implicit pledge."""
    match = re.fullmatch(
        r"(?:请)?(记下|暂停|恢复|完成|放下)共同计划[：:]\s*(.{1,120}?)[。！!]?",
        message.content.strip(),
    )
    if not match:
        return False
    operation, topic = match.groups()
    if not retainable(topic):
        raise DomainError("invalid_thread_topic", 422)
    status = {
        "记下": "open",
        "暂停": "paused",
        "恢复": "open",
        "完成": "resolved",
        "放下": "dropped",
    }[operation]
    thread = await session.scalar(
        select(SharedThread).where(SharedThread.pair_id == pair.id, SharedThread.topic == topic)
    )
    if operation == "记下":
        if thread:
            raise DomainError("thread_exists")
        if (
            await session.scalar(
                select(func.count())
                .select_from(SharedThread)
                .where(SharedThread.pair_id == pair.id)
            )
            >= 32
        ):
            raise DomainError("thread_limit", 422)
    else:
        if thread is None:
            raise DomainError("thread_not_found", 404)
        if status not in TRANSITIONS[thread.status]:
            raise DomainError("thread_transition_conflict")
        if (
            await session.scalar(
                select(func.count())
                .select_from(SharedThreadSource)
                .where(SharedThreadSource.thread_id == thread.id)
            )
            >= 32
        ):
            raise DomainError("thread_source_limit", 422)
    if (
        status == "open"
        and await session.scalar(
            select(func.count())
            .select_from(SharedThread)
            .where(SharedThread.pair_id == pair.id, SharedThread.status == "open")
        )
        >= 3
    ):
        raise DomainError("active_thread_limit", 422)
    if thread is None:
        thread = SharedThread(
            pair_id=pair.id, topic=topic, status="open", source_message_id=message.id
        )
        session.add(thread)
        await session.flush()
    else:
        thread.status, thread.revision = status, thread.revision + 1
    session.add(SharedThreadSource(thread_id=thread.id, message_id=message.id))
    task = await session.scalar(
        select(MemoryTask).where(MemoryTask.source_message_id == message.id)
    )
    if task:
        task.status = "done"
    pair.policy_revision += 1
    turn.policy_revision = pair.policy_revision
    turn.companion_plan = {**(turn.companion_plan or {}), "thread_control": status}
    return True
