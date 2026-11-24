"""Explicit source-backed shared plans, reusing existing transactions and forgetting."""

import re

from sqlalchemy import func, or_, select

from hmph.companion_policy import requested
from hmph.contracts import ACTIVE, DomainError
from hmph.dialogue import queue_dialogue
from hmph.models import MemoryTask, Message, Pair, SharedThread, SharedThreadSource, Turn
from hmph.persistence import append_event, terminate_turn
from hmph.source_policy import retainable
from hmph.working_memory import RawEvidence, query_terms

RETURNING = re.compile(r"我回来了|又见面|早上好|晚上好|^你好|^hello\b|^hi\b", re.I)
THREAD_QUERY = re.compile(
    r"(?:还有|还在|正在|什么|哪些|哪个|记得).{0,12}(?:共同计划|跟进.{0,8}计划)"
    r"|(?:共同计划|跟进.{0,8}计划).{0,8}(?:记得|还在|吗|么|呢)"
)
ENGLISH_THREAD_QUESTION = re.compile(
    r"^\s*(?:are we|do we|what|which)\b[^?]{0,80}\b(?:shared|joint)\s+plans?\b", re.I
)
NAMED_STATUS_QUESTION = re.compile(
    r"\s*(?P<topic>[^，,。！!？?；;、\n]{2,120}?)"
    r"(?:还在(?:继续|跟进)|现在(?:是)?什么状态|目前(?:是)?什么状态|暂停了|恢复了)"
    r"(?:吗|么|呢)?[？?]?\s*"
)
TRANSITIONS = {
    "open": {"paused", "resolved", "dropped"},
    "paused": {"open", "resolved", "dropped"},
    "resolved": set(),
    "dropped": set(),
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


async def prepare_threads(session, pair, turn, query, candidates):
    english_question = bool(ENGLISH_THREAD_QUESTION.search(query))
    thread_query = english_question or bool(THREAD_QUERY.search(query))
    question = query.rstrip()
    named_status = (
        NAMED_STATUS_QUESTION.fullmatch(question)
        if question.endswith(("吗", "么", "呢", "?", "？")) or "什么状态" in question
        else None
    )
    known = list(
        await session.scalars(
            select(SharedThread)
            .where(
                SharedThread.pair_id == pair.id,
                or_(
                    SharedThread.status == "open",
                    func.strpos(query.lower(), func.lower(SharedThread.topic)) > 0,
                    named_status is not None,
                ),
            )
            .order_by(SharedThread.id)
            .limit(32)
        )
    )
    threads = [thread for thread in known if thread.status == "open"]
    folded = query.casefold()
    terms = set(query_terms(query))
    matches = [thread for thread in known if thread.topic.casefold() in folded][:3]
    status_matches = _named_status_matches(named_status, known)
    if len(status_matches) == 1:
        matches = status_matches
    active_list = False
    if not matches:
        ranked = [(len(terms.intersection(query_terms(t.topic))), t) for t in threads]
        best = max((score for score, _ in ranked), default=0)
        nearest = [t for score, t in ranked if best and score == best]
        if len(nearest) == 1 and (best >= 2 or not thread_query):
            matches = nearest
        elif thread_query or RETURNING.search(query):
            matches = threads
            active_list = thread_query
    if active_list and candidates.raw:
        inactive = set(
            await session.scalars(
                select(SharedThreadSource.message_id)
                .join(SharedThread)
                .where(
                    SharedThread.pair_id == pair.id,
                    SharedThread.status != "open",
                    SharedThreadSource.message_id.in_([s.message_id for s in candidates.raw]),
                )
            )
        )
        candidates.raw[:] = [s for s in candidates.raw if s.message_id not in inactive]
    offered = []
    for thread in matches:
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
        existing = next((s for s in candidates.raw if s.message_id == source.id), None)
        if existing is not None:
            candidates.raw.remove(existing)
        else:
            if candidates.candidate_count >= candidates.plan.candidate_limit:
                continue
            candidates.candidate_count += 1
            origin = await session.get(Turn, source.turn_id)
            existing = RawEvidence(
                message_id=source.id,
                message_revision=source.revision,
                role="user",
                recorded_at=origin.created_at,
                content=thread.topic,
                start_offset=start,
                end_offset=start + len(thread.topic),
                truncated=True,
                source_timezone=origin.source_timezone,
            )
        candidates.raw.insert(len(offered), existing)
        offered.append(thread)
    no_active = (
        not matches and not threads and thread_query and (english_question or requested(query))
    )
    focus_sources = None
    if len(status_matches) == 1 and offered:
        thread = offered[0]
        focus_sources = set(
            await session.scalars(
                select(SharedThreadSource.message_id).where(
                    SharedThreadSource.thread_id == thread.id
                )
            )
        )
        focus_sources.add(thread.source_message_id)
        focus_sources.update(s.message_id for s in candidates.raw if thread.topic in s.content)
    return tuple(offered), active_list and len(offered) == len(threads), no_active, focus_sources


def _named_status_matches(match, known):
    if match is None:
        return []
    topic = re.sub(r"^(?:请问|咱们|我们)+", "", match["topic"]).strip(" 「」『』“”")
    topic = topic.replace("的", "")
    exact = [thread for thread in known if topic == thread.topic.replace("的", "")]
    if exact or len(topic) < 4:
        return exact
    return [thread for thread in known if thread.topic.replace("的", "").endswith(topic)]


async def apply_thread_command(session, pair, turn, message, *, proposed=None):
    """Only complete, explicit commands. No quote, hypothesis or implicit pledge."""
    match = proposed or re.fullmatch(
        r"(?:请)?(记下|暂停|恢复|完成|放下)共同计划[：:]\s*(.{1,120}?)[。！!]?",
        message.content.strip(),
    )
    if not match:
        return False
    operation, topic = match if proposed else match.groups()
    if not retainable(topic):
        raise DomainError("invalid_thread_topic", 422)
    status = {
        "记下": "open",
        "暂停": "paused",
        "恢复": "open",
        "完成": "resolved",
        "放下": "dropped",
        "进展": None,
    }[operation]
    thread = await session.scalar(
        select(SharedThread).where(SharedThread.pair_id == pair.id, SharedThread.topic == topic)
    )
    if operation == "进展":
        if thread is None or thread.status != "open":
            raise DomainError("thread_transition_conflict")
        status = thread.status
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
        if operation != "进展" and status not in TRANSITIONS[thread.status]:
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
        and operation != "进展"
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
    if task and proposed is None:
        task.status = "done"
    pair.policy_revision += 1
    turn.policy_revision = pair.policy_revision
    turn.companion_plan = {**(turn.companion_plan or {}), "thread_control": status}
    return True
