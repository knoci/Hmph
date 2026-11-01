"""Short-term state with a continuous source watermark and compare-and-set commits."""

from datetime import timedelta
from uuid import UUID, uuid5

from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.dialects.postgresql import insert

from hmph.contracts import DomainError
from hmph.database import Database
from hmph.dialogue_types import (
    DialogueView,
    GroundedText,
    OpenItem,
    ResponseMode,
    ShortTermContext,
    StateData,
    StateLease,
    StateMessage,
    StatePatch,
    StateResult,
)
from hmph.models import Chat, DialogueSource, DialogueState, DialogueTask, Message, Pair, Turn
from hmph.source_policy import retainable


async def queue_dialogue(session, turn, through_seq):
    # On an upgraded conversation, begin at this new turn, not a paid history backfill.
    await session.execute(
        insert(DialogueState)
        .values(
            conversation_id=turn.conversation_id,
            pair_id=turn.pair_id,
            revision=0,
            memory_epoch=turn.memory_epoch,
            coverage_from_seq=turn.source_seq,
            processed_through_seq=turn.source_seq - 1,
            data={},
        )
        .on_conflict_do_nothing()
    )
    stmt = insert(DialogueTask).values(
        conversation_id=turn.conversation_id,
        requested_through_seq=through_seq,
        model_mode=turn.model_mode,
        status="pending",
        fence=0,
        attempts=0,
    )
    await session.execute(
        stmt.on_conflict_do_update(
            index_elements=[DialogueTask.conversation_id],
            set_={
                "requested_through_seq": func.greatest(
                    DialogueTask.requested_through_seq, through_seq
                ),
                "status": case(
                    (DialogueTask.status.in_(("idle", "failed")), "pending"),
                    else_=DialogueTask.status,
                ),
                "model_mode": turn.model_mode,
                "attempts": case(
                    (DialogueTask.status.in_(("idle", "failed")), 0), else_=DialogueTask.attempts
                ),
            },
        )
    )


def state_sources(data: StateData):
    anchors = [data.topic, data.goal, *data.entities]
    for item in data.open_items:
        anchors.extend([item.content, item.transition_evidence])
    if data.response_mode:
        anchors.append(data.response_mode.evidence)
    return {anchor.message_id: anchor for anchor in anchors if anchor is not None}


async def short_term_context(session, turn, budget=1300):
    state = await session.get(DialogueState, turn.conversation_id)
    if state is None or state.memory_epoch != turn.memory_epoch:
        return ShortTermContext()
    data = StateData.model_validate(state.data)
    result = ShortTermContext(
        revision=state.revision,
        processed_through_seq=state.processed_through_seq,
        coverage_from_seq=state.coverage_from_seq,
    )

    def fits(candidate):
        return len(candidate.model_dump_json().encode()) <= budget

    atoms = [("topic", data.topic), ("goal", data.goal)]
    if data.response_mode and turn.source_seq <= data.response_mode.expires_after_seq:
        atoms.append(("response_mode", data.response_mode))
    for field, value in atoms:
        if value is not None:
            proposed = result.model_copy(deep=True)
            setattr(proposed.data, field, value)
            if fits(proposed):
                result = proposed
            else:
                result.state_truncated = True
    for field, values in (("open_items", data.open_items), ("entities", data.entities)):
        for value in reversed(values):
            proposed = result.model_copy(deep=True)
            getattr(proposed.data, field).insert(0, value)
            if fits(proposed):
                result = proposed
            else:
                result.state_truncated = True
    return result


async def invalidate_dialogue(session, pair, removed_messages):
    affected = set(
        await session.scalars(
            select(DialogueSource.conversation_id).where(
                DialogueSource.message_id.in_(removed_messages)
            )
        )
    )
    states = await session.scalars(select(DialogueState).where(DialogueState.pair_id == pair.id))
    for state in states:
        state.memory_epoch = pair.memory_epoch
        task = await session.get(DialogueTask, state.conversation_id, with_for_update=True)
        task.fence += 1
        task.worker_id, task.lease_until, task.attempts = None, None, 0
        if state.conversation_id in affected:
            state.data = {}
            state.revision += 1
            state.processed_through_seq = state.coverage_from_seq - 1
            await session.execute(
                delete(DialogueSource).where(
                    DialogueSource.conversation_id == state.conversation_id
                )
            )
        task.status = (
            "pending" if task.requested_through_seq > state.processed_through_seq else "idle"
        )
        task.error_code = None
    return len(affected)


class Dialogue:
    def __init__(self, db: Database):
        self.db = db

    async def claim(self, worker_id, lease_seconds=30, *, model_mode="fake"):
        if not 1 <= len(worker_id) <= 100 or lease_seconds <= 0:
            raise ValueError("Invalid state lease")
        async with self.db.sessions.begin() as session:
            pair = await session.scalar(
                select(Pair)
                .join(DialogueState)
                .join(DialogueTask)
                .where(
                    DialogueTask.status == "pending",
                    DialogueTask.model_mode == model_mode,
                )
                .order_by(DialogueTask.conversation_id)
                .limit(1)
                .with_for_update(of=Pair, skip_locked=True)
            )
            if pair is None:
                return None
            task = await session.scalar(
                select(DialogueTask)
                .join(DialogueState)
                .where(
                    DialogueState.pair_id == pair.id,
                    DialogueTask.status == "pending",
                    DialogueTask.model_mode == model_mode,
                )
                .order_by(DialogueTask.conversation_id)
                .limit(1)
                .with_for_update(of=DialogueTask)
            )
            if task is None:
                return None
            state = await session.get(DialogueState, task.conversation_id)
            through = min(task.requested_through_seq, state.processed_through_seq + 32)
            rows = await session.scalars(
                select(Message)
                .join(Turn)
                .where(
                    Message.conversation_id == state.conversation_id,
                    Message.source_seq > state.processed_through_seq,
                    Message.source_seq <= through,
                    Message.deleted.is_(False),
                    or_(Message.role == "user", Turn.status == "completed"),
                )
                .order_by(Message.source_seq)
            )
            chosen, used = [], len(StateData.model_validate(state.data).model_dump_json().encode())
            for row in rows:
                size = len(row.content.encode()) + 160
                if used + size > 12000:
                    if not chosen:
                        task.status, task.error_code = "failed", "state_input_budget"
                        return None
                    through = row.source_seq - 1
                    break
                if retainable(row.content):
                    chosen.append(StateMessage.model_validate(row, from_attributes=True))
                used += size
            now = await session.scalar(select(func.clock_timestamp()))
            task.status, task.worker_id, task.error_code = "leased", worker_id, None
            task.fence += 1
            task.attempts += 1
            task.lease_until = now + timedelta(seconds=lease_seconds)
            return StateLease(
                conversation_id=state.conversation_id,
                pair_id=pair.id,
                memory_epoch=pair.memory_epoch,
                revision=state.revision,
                through_seq=through,
                worker_id=worker_id,
                fence=task.fence,
                data=StateData.model_validate(state.data),
                messages=tuple(chosen),
            )

    async def _lease(self, session, lease):
        pair = await session.scalar(select(Pair).where(Pair.id == lease.pair_id).with_for_update())
        state = await session.get(DialogueState, lease.conversation_id)
        task = await session.get(DialogueTask, lease.conversation_id, with_for_update=True)
        now = await session.scalar(select(func.clock_timestamp()))
        if (
            pair is None
            or state is None
            or task is None
            or state.pair_id != pair.id
            or task.status != "leased"
            or task.worker_id != lease.worker_id
            or task.fence != lease.fence
            or task.lease_until is None
            or task.lease_until <= now
            or pair.memory_epoch != lease.memory_epoch
            or state.revision != lease.revision
        ):
            raise DomainError("stale_state_lease")
        return pair, state, task, now

    async def renew(self, lease, lease_seconds=30):
        if lease_seconds <= 0:
            raise ValueError("Invalid state lease")
        async with self.db.sessions.begin() as session:
            _, _, task, now = await self._lease(session, lease)
            task.lease_until = now + timedelta(seconds=lease_seconds)

    async def fail(self, lease, *, code="state_failed", retryable=False, usage=None):
        if code not in {"state_failed", "unsupported_state", "provider_failed", "state_timeout"}:
            code = "state_failed"
        async with self.db.sessions.begin() as session:
            try:
                _, _, task, _ = await self._lease(session, lease)
            except DomainError:
                return
            task.status = "pending" if retryable and task.attempts < 3 else "failed"
            task.error_code, task.worker_id, task.lease_until = code, None, None
            if usage is not None:
                task.usage = usage
            task.fence += 1

    async def recover_expired(self, limit=100):
        recovered = 0
        for _ in range(limit):
            async with self.db.sessions.begin() as session:
                now = await session.scalar(select(func.clock_timestamp()))
                pair = await session.scalar(
                    select(Pair)
                    .join(DialogueState)
                    .join(DialogueTask)
                    .where(
                        DialogueTask.status == "leased",
                        DialogueTask.lease_until <= now,
                    )
                    .order_by(DialogueTask.conversation_id)
                    .limit(1)
                    .with_for_update(of=Pair, skip_locked=True)
                )
                if pair is None:
                    break
                task = await session.scalar(
                    select(DialogueTask)
                    .join(DialogueState)
                    .where(
                        DialogueState.pair_id == pair.id,
                        DialogueTask.status == "leased",
                        DialogueTask.lease_until <= func.clock_timestamp(),
                    )
                    .order_by(DialogueTask.conversation_id)
                    .limit(1)
                    .with_for_update(of=DialogueTask)
                )
                if task is None:
                    continue
                task.status = "pending" if task.attempts < 3 else "failed"
                task.error_code, task.worker_id, task.lease_until = "worker_lost", None, None
                task.fence += 1
                recovered += 1
        return recovered

    async def commit(self, lease: StateLease, patch: StatePatch, *, usage=None):
        async with self.db.sessions.begin() as session:
            try:
                _, state, task, _ = await self._lease(session, lease)
            except DomainError:
                return StateResult(status="stale")
            messages = {row.id: row for row in lease.messages}

            async def ground(anchor):
                source = messages.get(anchor.message_id)
                row = await session.get(Message, anchor.message_id) if source else None
                if (
                    row is None
                    or row.deleted
                    or row.revision != source.revision
                    or row.role != "user"
                    or anchor.quote not in row.content
                    or not anchor.text.strip()
                    or "\x00" in anchor.text
                    or "\x00" in anchor.quote
                    or not retainable(row.content)
                ):
                    raise DomainError("unsupported_state", 422)
                # Do not persist a condensed label that is absent from the source.
                grounded_text = anchor.text if anchor.text in anchor.quote else anchor.quote
                if len(grounded_text) > 120:
                    raise DomainError("unsupported_state", 422)
                return GroundedText(
                    **{**anchor.model_dump(), "text": grounded_text},
                    message_revision=row.revision,
                    source_seq=row.source_seq,
                )

            data = StateData.model_validate(state.data)
            if patch.topic:
                topic = await ground(patch.topic)
                if data.topic and data.topic.text != topic.text:
                    data.goal, data.response_mode, data.entities = None, None, []
                data.topic = topic
            if patch.goal:
                data.goal = await ground(patch.goal)
            for anchor in patch.entities:
                entity = await ground(anchor)
                data.entities = [v for v in data.entities if v.text != entity.text] + [entity]
            data.entities = data.entities[-6:]
            for anchor in patch.open_items:
                content = await ground(anchor)
                item_id = uuid5(lease.conversation_id, str(content.message_id) + content.text)
                if not any(item.id == item_id for item in data.open_items):
                    data.open_items.append(OpenItem(id=item_id, content=content))
            for transition in patch.transitions:
                item = next((v for v in data.open_items if v.id == transition.item_id), None)
                if item is None:
                    raise DomainError("unknown_state_item", 422)
                item.transition_evidence = await ground(transition.evidence)
                item.status = transition.status
                if data.goal and data.goal.text == item.content.text:
                    data.goal = None
            if patch.response_mode:
                evidence = await ground(patch.response_mode.evidence)
                data.response_mode = ResponseMode(
                    mode=patch.response_mode.mode,
                    evidence=evidence,
                    expires_after_seq=evidence.source_seq + 6,
                )
            # Do not silently drop unresolved items to make a summary fit.
            opened = [item for item in data.open_items if item.status == "open"]
            closed = [item for item in data.open_items if item.status != "open"][-4:]
            data.open_items = opened + closed
            if len(opened) > 12 or len(data.model_dump_json().encode()) > 10000:
                raise DomainError("state_capacity", 422)
            state.data = data.model_dump(mode="json")
            state.revision += 1
            state.processed_through_seq = lease.through_seq
            state.memory_epoch = lease.memory_epoch
            await session.execute(
                delete(DialogueSource).where(
                    DialogueSource.conversation_id == state.conversation_id
                )
            )
            for anchor in state_sources(data).values():
                session.add(
                    DialogueSource(
                        conversation_id=state.conversation_id,
                        message_id=anchor.message_id,
                        message_revision=anchor.message_revision,
                    )
                )
            task.status = "pending" if task.requested_through_seq > lease.through_seq else "idle"
            task.worker_id, task.lease_until, task.error_code, task.usage = None, None, None, usage
            task.attempts = 0
            return StateResult(status="committed")

    async def read(self, owner_id: UUID, conversation_id: UUID):
        async with self.db.sessions() as session:
            chat = await session.scalar(
                select(Chat)
                .join(Pair)
                .where(
                    Chat.id == conversation_id,
                    Pair.owner_id == owner_id,
                )
            )
            if chat is None:
                raise DomainError("not_found", 404)
            state = await session.get(DialogueState, conversation_id)
            task = await session.get(DialogueTask, conversation_id)
            watermark = state.processed_through_seq if state else 0
            pending = await session.scalar(
                select(func.count())
                .select_from(Message)
                .join(Turn)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.source_seq > watermark,
                    Message.deleted.is_(False),
                    or_(Message.role == "user", Turn.status == "completed"),
                )
            )
            return DialogueView(
                conversation_id=conversation_id,
                revision=state.revision if state else 0,
                processed_through_seq=watermark,
                coverage_from_seq=state.coverage_from_seq if state else 1,
                data=StateData.model_validate(state.data) if state else StateData(),
                pending_messages=pending,
                task_status=task.status if task else "idle",
                error_code=task.error_code if task else None,
                usage=task.usage if task else None,
            )
