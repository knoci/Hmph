import hashlib
import hmac
import json
import re
from datetime import timedelta
from uuid import UUID, uuid5

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert

from hmph.contracts import (
    ACTIVE,
    ChatView,
    DomainError,
    EventView,
    Identity,
    Lease,
    MessageView,
    TurnView,
)
from hmph.database import Database
from hmph.model_gateway import ChatMessage, GenerationRequest
from hmph.models import Character, Chat, Job, Message, Owner, Pair, StreamEvent, Turn, WriteRequest

DEFAULT_OWNER = UUID("550062e6-03a7-4ff0-bb47-c5eef6515a18")
DEFAULT_CHARACTER = UUID("af1d4af8-7c4e-49b3-9b42-83d5457bedce")


class Conversation:
    def __init__(self, db: Database, *, digest_key: bytes, turn_timeout: float = 45):
        self.db = db
        self.digest_key = digest_key
        self.turn_timeout = turn_timeout

    async def bootstrap(self, owner_id: UUID = DEFAULT_OWNER) -> Identity:
        async with self.db.sessions.begin() as session:
            await session.execute(
                insert(Owner).values(id=owner_id, name="Local owner").on_conflict_do_nothing()
            )
            await session.execute(
                insert(Character)
                .values(
                    id=DEFAULT_CHARACTER,
                    slug="hmph",
                    persona_version=1,
                )
                .on_conflict_do_nothing()
            )
        return Identity(owner_id=owner_id, character_id=DEFAULT_CHARACTER)

    def _digest(self, target: str, body: dict) -> str:
        raw = json.dumps([target, body], sort_keys=True, ensure_ascii=False).encode()
        return hmac.new(self.digest_key, raw, hashlib.sha256).hexdigest()

    @staticmethod
    def _request_id(request_id: str):
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
            raise DomainError("invalid_request_id", 422)

    async def _owner_lock(self, session, owner_id):
        if (
            await session.scalar(select(Owner).where(Owner.id == owner_id).with_for_update())
            is None
        ):
            raise DomainError("not_found", 404)

    async def _request(self, session, owner_id, operation, request_id, digest):
        self._request_id(request_id)
        existing = await session.get(WriteRequest, (owner_id, operation, request_id))
        if existing and not hmac.compare_digest(existing.digest, digest):
            raise DomainError("idempotency_conflict")
        return existing.result_id if existing else None

    @staticmethod
    def _save_request(session, owner_id, operation, request_id, digest, result_id):
        session.add(
            WriteRequest(
                owner_id=owner_id,
                operation=operation,
                request_id=request_id,
                digest=digest,
                result_id=result_id,
            )
        )

    async def create_chat(self, owner_id: UUID, character_id: UUID, request_id: str) -> ChatView:
        digest = self._digest("conversations", {"character_id": str(character_id)})
        async with self.db.sessions.begin() as session:
            await self._owner_lock(session, owner_id)
            existing = await self._request(session, owner_id, "chat.create", request_id, digest)
            if existing:
                return ChatView.model_validate(await session.get(Chat, existing))
            if await session.get(Character, character_id) is None:
                raise DomainError("not_found", 404)
            pair_id = uuid5(owner_id, str(character_id))
            await session.execute(
                insert(Pair)
                .values(
                    id=pair_id,
                    owner_id=owner_id,
                    character_id=character_id,
                    preferences={},
                    policy_revision=1,
                    memory_epoch=1,
                )
                .on_conflict_do_nothing()
            )
            chat = Chat(pair_id=pair_id)
            session.add(chat)
            await session.flush()
            self._save_request(session, owner_id, "chat.create", request_id, digest, chat.id)
            return ChatView.model_validate(chat)

    async def _chat(self, session, owner_id, chat_id):
        chat = await session.scalar(
            select(Chat)
            .join(Pair)
            .where(
                Chat.id == chat_id,
                Pair.owner_id == owner_id,
            )
        )
        if chat is None:
            raise DomainError("not_found", 404)
        return chat

    async def _turn(self, session, owner_id, turn_id):
        turn = await session.scalar(
            select(Turn)
            .join(Pair)
            .where(
                Turn.id == turn_id,
                Pair.owner_id == owner_id,
            )
        )
        if turn is None:
            raise DomainError("not_found", 404)
        return turn

    async def _locked_turn(self, session, owner_id, turn_id):
        pair = await session.scalar(
            select(Pair)
            .join(Turn, Turn.pair_id == Pair.id)
            .where(
                Turn.id == turn_id,
                Pair.owner_id == owner_id,
            )
            .with_for_update(of=Pair)
        )
        if pair is None:
            raise DomainError("not_found", 404)
        turn = await session.scalar(select(Turn).where(Turn.id == turn_id).with_for_update())
        return pair, turn

    @staticmethod
    def _event(session, turn, event_type, payload, *, segment_index=None):
        event = StreamEvent(
            turn_id=turn.id,
            id=turn.next_event_id,
            type=event_type,
            payload=payload,
            segment_index=segment_index,
        )
        turn.next_event_id += 1
        session.add(event)
        return event

    async def accept(
        self,
        owner_id: UUID,
        chat_id: UUID,
        request_id: str,
        content: str,
        *,
        model_mode: str = "fake",
    ) -> TurnView:
        if model_mode not in ("fake", "deepseek"):
            raise DomainError("invalid_model_mode", 422)
        if not content.strip() or "\x00" in content or len(content) > 4000:
            raise DomainError("invalid_input", 422)
        digest = self._digest(str(chat_id), {"content": content})
        async with self.db.sessions.begin() as session:
            await self._owner_lock(session, owner_id)
            chat = await self._chat(session, owner_id, chat_id)
            existing = await self._request(session, owner_id, "turn.create", request_id, digest)
            if existing:
                return TurnView.model_validate(await self._turn(session, owner_id, existing))
            pair = await session.scalar(
                select(Pair).where(Pair.id == chat.pair_id).with_for_update()
            )
            active = await session.scalar(
                select(Turn.id)
                .where(
                    Turn.pair_id == pair.id,
                    Turn.status.in_(ACTIVE),
                )
                .limit(1)
            )
            if active:
                raise DomainError("turn_in_progress")
            now = await session.scalar(select(func.clock_timestamp()))
            turn = Turn(
                conversation_id=chat.id,
                pair_id=pair.id,
                source_seq=chat.next_source_seq,
                policy_revision=pair.policy_revision,
                memory_epoch=pair.memory_epoch,
                deadline=now + timedelta(seconds=self.turn_timeout),
                model_mode=model_mode,
            )
            chat.next_source_seq += 2
            session.add(turn)
            await session.flush()
            session.add(
                Message(
                    conversation_id=chat.id,
                    turn_id=turn.id,
                    source_seq=turn.source_seq,
                    role="user",
                    content=content,
                )
            )
            session.add(Job(turn_id=turn.id))
            self._event(session, turn, "turn.accepted", {"model_mode": model_mode})
            self._save_request(session, owner_id, "turn.create", request_id, digest, turn.id)
            await session.flush()
            return TurnView.model_validate(turn)

    async def get_turn(self, owner_id: UUID, turn_id: UUID) -> TurnView:
        async with self.db.sessions() as session:
            return TurnView.model_validate(await self._turn(session, owner_id, turn_id))

    async def chats(self, owner_id: UUID) -> list[ChatView]:
        async with self.db.sessions() as session:
            rows = await session.scalars(
                select(Chat)
                .join(Pair)
                .where(
                    Pair.owner_id == owner_id,
                )
                .order_by(Chat.created_at.desc(), Chat.id)
                .limit(100)
            )
            return [ChatView.model_validate(row) for row in rows]

    async def messages(self, owner_id: UUID, chat_id: UUID) -> list[MessageView]:
        async with self.db.sessions() as session:
            await self._chat(session, owner_id, chat_id)
            rows = await session.scalars(
                select(Message)
                .where(
                    Message.conversation_id == chat_id,
                )
                .order_by(Message.source_seq)
            )
            return [MessageView.model_validate(row) for row in rows]

    async def events(self, owner_id: UUID, turn_id: UUID, after: int = 0) -> list[EventView]:
        async with self.db.sessions() as session:
            turn = await self._turn(session, owner_id, turn_id)
            if after < 0 or after >= turn.next_event_id:
                raise DomainError("invalid_cursor", 422)
            rows = await session.scalars(
                select(StreamEvent)
                .where(
                    StreamEvent.turn_id == turn_id,
                    StreamEvent.id > after,
                )
                .order_by(StreamEvent.id)
                .limit(200)
            )
            return [EventView.model_validate(row) for row in rows]

    async def _terminal(self, session, turn, status, error_code=None):
        if turn.status not in ACTIVE:
            return
        turn.status = status
        turn.error_code = error_code
        job = await session.scalar(select(Job).where(Job.turn_id == turn.id).with_for_update())
        job.status = "done"
        job.fence += 1
        job.lease_until = None
        job.worker_id = None
        self._event(
            session, turn, f"turn.{status}", {"error_code": error_code, "usage": turn.usage}
        )

    async def cancel(self, owner_id: UUID, turn_id: UUID, request_id: str) -> TurnView:
        digest = self._digest(str(turn_id), {"action": "cancel"})
        async with self.db.sessions.begin() as session:
            await self._owner_lock(session, owner_id)
            _, turn = await self._locked_turn(session, owner_id, turn_id)
            existing = await self._request(session, owner_id, "turn.cancel", request_id, digest)
            if not existing:
                await self._terminal(session, turn, "cancelled")
                self._save_request(session, owner_id, "turn.cancel", request_id, digest, turn.id)
            await session.flush()
            return TurnView.model_validate(turn)

    async def claim(
        self, worker_id: str, lease_seconds: float = 15, *, model_mode: str = "fake"
    ) -> Lease | None:
        if lease_seconds <= 0 or not 1 <= len(worker_id) <= 100:
            raise ValueError("Invalid worker lease settings")
        async with self.db.sessions.begin() as session:
            pair = await session.scalar(
                select(Pair)
                .join(Turn, Turn.pair_id == Pair.id)
                .join(Job, Job.turn_id == Turn.id)
                .where(
                    Job.status == "pending",
                    Turn.status == "accepted",
                    Turn.model_mode == model_mode,
                )
                .order_by(Turn.created_at, Turn.id)
                .limit(1)
                .with_for_update(of=Pair, skip_locked=True)
            )
            if pair is None:
                return None
            turn = await session.scalar(
                select(Turn)
                .where(
                    Turn.pair_id == pair.id,
                    Turn.status == "accepted",
                )
                .with_for_update()
            )
            # The joined queue row can change before the pair lock is acquired.
            if turn is None:
                return None
            now = await session.scalar(select(func.clock_timestamp()))
            if now >= turn.deadline:
                await self._terminal(session, turn, "failed", "turn_timeout")
                return None
            job = await session.scalar(select(Job).where(Job.turn_id == turn.id).with_for_update())
            if job.status != "pending":
                return None
            if (
                turn.policy_revision != pair.policy_revision
                or turn.memory_epoch != pair.memory_epoch
            ):
                await self._terminal(session, turn, "invalidated", "snapshot_changed")
                return None
            job.status = "leased"
            job.fence += 1
            job.worker_id = worker_id
            job.lease_until = min(now + timedelta(seconds=lease_seconds), turn.deadline)
            turn.status = "generating"
            content = await session.scalar(
                select(Message.content).where(
                    Message.turn_id == turn.id,
                    Message.role == "user",
                )
            )
            return Lease(
                turn_id=turn.id,
                owner_id=pair.owner_id,
                worker_id=worker_id,
                fence=job.fence,
                policy_revision=pair.policy_revision,
                memory_epoch=pair.memory_epoch,
                deadline=turn.deadline,
                remaining_seconds=(turn.deadline - now).total_seconds(),
                user_content=content,
            )

    async def _lease(self, session, lease):
        pair, turn = await self._locked_turn(session, lease.owner_id, lease.turn_id)
        job = await session.scalar(select(Job).where(Job.turn_id == turn.id).with_for_update())
        now = await session.scalar(select(func.clock_timestamp()))
        if (
            turn.status != "generating"
            or job.status != "leased"
            or job.fence != lease.fence
            or job.worker_id != lease.worker_id
            or job.lease_until is None
            or job.lease_until <= now
            or turn.deadline <= now
            or pair.policy_revision != lease.policy_revision
            or pair.memory_epoch != lease.memory_epoch
        ):
            raise DomainError("stale_lease")
        return turn, job, now

    async def renew(self, lease: Lease, lease_seconds: float):
        async with self.db.sessions.begin() as session:
            turn, job, now = await self._lease(session, lease)
            job.lease_until = min(now + timedelta(seconds=lease_seconds), turn.deadline)

    async def publish(self, lease: Lease, segment_index: int, content: str) -> EventView:
        if not content or len(content) > 256 or "\x00" in content:
            raise DomainError("invalid_segment", 422)
        async with self.db.sessions.begin() as session:
            turn, _, _ = await self._lease(session, lease)
            if segment_index < turn.published_segments:
                event = await session.scalar(
                    select(StreamEvent).where(
                        StreamEvent.turn_id == turn.id,
                        StreamEvent.segment_index == segment_index,
                    )
                )
                if event is None or event.payload["text"] != content:
                    raise DomainError("segment_conflict")
                return EventView.model_validate(event)
            if segment_index != turn.published_segments or segment_index >= 128:
                raise DomainError("segment_order")
            message = await session.scalar(
                select(Message).where(
                    Message.turn_id == turn.id,
                    Message.role == "assistant",
                )
            )
            if message is None:
                message = Message(
                    conversation_id=turn.conversation_id,
                    turn_id=turn.id,
                    source_seq=turn.source_seq + 1,
                    role="assistant",
                    content="",
                )
                session.add(message)
            if len(message.content) + len(content) > 2000:
                raise DomainError("output_limit")
            message.content += content
            event = self._event(
                session, turn, "text.delta", {"text": content}, segment_index=segment_index
            )
            turn.published_segments += 1
            await session.flush()
            return EventView.model_validate(event)

    async def finish(
        self, lease: Lease, *, error_code: str | None = None, usage: dict | None = None
    ) -> TurnView:
        async with self.db.sessions.begin() as session:
            turn, _, _ = await self._lease(session, lease)
            turn.usage = usage
            await self._terminal(session, turn, "failed" if error_code else "completed", error_code)
            await session.flush()
            return TurnView.model_validate(turn)

    async def generation_request(self, lease: Lease) -> GenerationRequest:
        async with self.db.sessions.begin() as session:
            turn, _, _ = await self._lease(session, lease)
            rows = list(
                await session.scalars(
                    select(Message)
                    .join(Turn, Message.turn_id == Turn.id)
                    .where(
                        Message.conversation_id == turn.conversation_id,
                        Message.source_seq <= turn.source_seq,
                        or_(Message.role == "user", Turn.status == "completed"),
                    )
                    .order_by(Message.source_seq.desc())
                    .limit(40)
                )
            )
            chosen = []
            remaining = 6000  # Conservative UTF-8 budget, not a claimed exact tokenizer.
            for row in rows:
                size = len(row.content.encode("utf-8")) + 64
                if size > remaining:
                    if not chosen:
                        raise DomainError("input_budget_exceeded", 422)
                    break
                chosen.append(ChatMessage(role=row.role, content=row.content))
                remaining -= size
            chosen.reverse()
            if chosen and chosen[0].role == "assistant":
                chosen.pop(0)
            return GenerationRequest(messages=tuple(chosen))

    async def recover_expired(self, limit: int = 100) -> int:
        recovered = 0
        for _ in range(limit):
            async with self.db.sessions.begin() as session:
                now = await session.scalar(select(func.clock_timestamp()))
                pair = await session.scalar(
                    select(Pair)
                    .join(Turn, Turn.pair_id == Pair.id)
                    .join(Job, Job.turn_id == Turn.id)
                    .where(
                        Turn.status.in_(ACTIVE),
                        or_(
                            Turn.deadline <= now,
                            and_(Job.status == "leased", Job.lease_until <= now),
                        ),
                    )
                    .order_by(Turn.created_at)
                    .limit(1)
                    .with_for_update(of=Pair, skip_locked=True)
                )
                if pair is None:
                    break
                turn = await session.scalar(
                    select(Turn)
                    .where(
                        Turn.pair_id == pair.id,
                        Turn.status.in_(ACTIVE),
                    )
                    .with_for_update()
                )
                if turn is None:
                    continue
                job = await session.scalar(
                    select(Job).where(Job.turn_id == turn.id).with_for_update()
                )
                now = await session.scalar(select(func.clock_timestamp()))
                if turn.deadline > now and (
                    job.status != "leased" or job.lease_until is None or job.lease_until > now
                ):
                    continue
                reason = "turn_timeout" if turn.deadline <= now else "worker_lost"
                await self._terminal(session, turn, "failed", reason)
                recovered += 1
        return recovered
