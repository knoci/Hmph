"""Explicit settings share the message/source/forget lifecycle, without model jobs."""

from sqlalchemy import func, select

from hmph.contracts import ACTIVE, DomainError
from hmph.dialogue import queue_dialogue
from hmph.models import MemoryEntry, MemorySource, MemoryTask, MemoryVersion, Message, Pair, Turn
from hmph.persistence import append_event, terminate_turn
from hmph.preference_types import expression_preferences


class Preferences:
    def __init__(self, conversation):
        self.conversation = conversation
        self.db = conversation.db
        self.ledger = conversation.ledger

    @staticmethod
    def view(pair):
        return {
            "revision": pair.policy_revision if pair else 1,
            "preferences": expression_preferences(pair.preferences if pair else {}).model_dump(),
            "source_memory_id": pair.preferences.get("_source_memory_id") if pair else None,
        }

    async def read(self, owner_id, character_id):
        async with self.db.sessions() as session:
            pair = await session.scalar(
                select(Pair).where(Pair.owner_id == owner_id, Pair.character_id == character_id)
            )
            return self.view(pair)

    async def update(
        self, owner_id, chat_id, request_id, expected_revision, preferences, *, model_mode="fake"
    ):
        digest = self.ledger.digest(
            str(chat_id),
            {"expected_revision": expected_revision, "preferences": preferences.model_dump()},
        )
        async with self.db.sessions.begin() as session:
            await self.ledger.owner_lock(session, owner_id)
            chat = await self.conversation._chat(session, owner_id, chat_id)
            pair = await session.scalar(
                select(Pair).where(Pair.id == chat.pair_id).with_for_update()
            )
            existing = await self.ledger.request(
                session, owner_id, "preferences.update", request_id, digest
            )
            if existing:
                return self.view(pair)
            if pair.policy_revision != expected_revision:
                raise DomainError("preference_revision_conflict")
            boundaries = {
                (b["kind"], b["target"]): b for b in pair.preferences.get("_boundaries", [])
            }
            previous_nickname = pair.preferences.get("nickname")
            if preferences.nickname:
                boundaries.pop(("nickname", preferences.nickname), None)
            if (
                previous_nickname
                and preferences.nickname != previous_nickname
                and len(boundaries) >= 16
            ):
                raise DomainError("companion_control_capacity", 422)
            active = await session.scalars(
                select(Turn)
                .where(Turn.pair_id == pair.id, Turn.status.in_(ACTIVE))
                .with_for_update()
            )
            for turn in active:
                await terminate_turn(session, turn, "invalidated", "preferences_changed")
            previous = pair.preferences.get("_source_memory_id")
            if previous:
                old = await session.get(MemoryEntry, previous)
                if old and old.status == "active":
                    old.scope = "historical"
            now = await session.scalar(select(func.clock_timestamp()))
            pair.policy_revision += 1
            turn = Turn(
                conversation_id=chat.id,
                pair_id=pair.id,
                status="completed",
                source_seq=chat.next_source_seq,
                policy_revision=pair.policy_revision,
                memory_epoch=pair.memory_epoch,
                source_timezone=self.conversation.source_timezone,
                deadline=now,
                created_at=now,
                model_mode=model_mode,
            )
            chat.next_source_seq += 2
            session.add(turn)
            await session.flush()
            nickname = (
                f"称呼我为“{preferences.nickname}”" if preferences.nickname else "暂不指定称呼"
            )
            length = {
                "short": "回复简短一些",
                "normal": "保持自然回复长度",
                "long": "回复可以详细一些",
            }
            play = {"low": "少开玩笑", "normal": "可以自然打趣"}
            support = {
                "listen": "先听我说，少提供建议",
                "ask_first": "给建议前先问我",
                "suggest": "可以主动提供建议",
            }
            content = (
                "用户明确设置表达偏好："
                + "；".join(
                    (
                        nickname,
                        length[preferences.reply_length],
                        play[preferences.playfulness],
                        support[preferences.support_mode],
                    )
                )
                + "。"
            )
            message = Message(
                conversation_id=chat.id,
                turn_id=turn.id,
                source_seq=turn.source_seq,
                role="user",
                content=content,
            )
            session.add(message)
            await session.flush()
            entry = MemoryEntry(
                pair_id=pair.id,
                kind="fact",
                predicate="expression_preferences",
                scope="current",
                revision=1,
                source_order=pair.next_memory_seq,
            )
            session.add(entry)
            await session.flush()
            session.add(MemoryVersion(memory_id=entry.id, revision=1, value=content))
            await session.flush()
            session.add(
                MemorySource(
                    memory_id=entry.id,
                    revision=1,
                    message_id=message.id,
                    message_revision=1,
                    quote=content,
                )
            )
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
            if previous_nickname and preferences.nickname != previous_nickname:
                boundaries[("nickname", previous_nickname)] = {
                    "target": previous_nickname,
                    "kind": "nickname",
                    "mode": "never",
                    "source_memory_id": str(entry.id),
                }
            pair.preferences = {
                **preferences.model_dump(),
                "_source_memory_id": str(entry.id),
                "_field_sources": dict.fromkeys(type(preferences).model_fields, str(entry.id)),
                "_boundaries": list(boundaries.values()),
            }
            append_event(session, turn, "turn.completed", {"operation": "preferences.updated"})
            await queue_dialogue(session, turn, turn.source_seq + 1)
            self.ledger.save(session, owner_id, "preferences.update", request_id, digest, entry.id)
            return self.view(pair)
