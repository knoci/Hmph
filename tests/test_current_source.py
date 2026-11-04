import json

import httpx
from pydantic import SecretStr
from sqlalchemy import select

from hmph.companion import checked_mentions
from hmph.deepseek import DeepSeekModel
from hmph.memory import Memory
from hmph.model_gateway import ModelDelta, ModelUsage, SourceMention
from hmph.models import CompanionMention, Turn
from hmph.worker import Worker


class CurrentReply:
    async def stream(self, request):
        yield ModelDelta(
            text="那就去远山公司。",
            usage=ModelUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


async def current_turn(conversation):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "current-scope")
    turn = await conversation.accept(
        identity.owner_id,
        chat.id,
        "current",
        "原来北岸工作室的面试取消，改成明天去远山公司。我现在要去哪家公司？",
        model_mode="deepseek",
    )
    source = next(
        m for m in await conversation.messages(identity.owner_id, chat.id) if m.turn_id == turn.id
    )
    hint = SourceMention(source_id=source.id, source_quote="改成明天去远山公司", quote="远山公司")
    return identity, chat, turn, source, hint


async def test_acknowledgment_of_current_correction_is_not_old_memory_use(conversation):
    identity, chat, turn, source, hint = await current_turn(conversation)

    def provider(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert payload["current_user_source"] == str(source.id)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "allow": True,
                                    "reason": "ok",
                                    "mentions": [hint.model_dump(mode="json")],
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    review_model = DeepSeekModel(
        SecretStr("fixture-provider-key"), transport=httpx.MockTransport(provider)
    )

    async def planner(request):
        raise AssertionError("A current correction must not incur a planning call")

    conversation.recall_planner = planner
    worker = Worker(
        conversation, CurrentReply(), model_mode="deepseek", reviewer=review_model.review
    )
    await worker.run_once()
    saved = await conversation.get_turn(identity.owner_id, turn.id)
    assert saved.status == "completed" and saved.usage["guard_action"] == "annotation_only"
    assert saved.usage["planner"]["attempts"] == 0
    messages = await conversation.messages(identity.owner_id, chat.id)
    assert next(m.content for m in messages if m.role == "assistant") == "那就去远山公司。"
    async with conversation.db.sessions() as session:
        mentions = list(
            await session.scalars(
                select(CompanionMention).where(
                    CompanionMention.turn_id == turn.id,
                )
            )
        )
        assert [m.source_message_id for m in mentions] == [source.id]
    memory = Memory(conversation.db, digest_key=conversation.digest_key)
    receipt = await memory.forget_source(
        identity.owner_id, source.id, "forget-current", expected_revision=source.revision
    )
    assert receipt.status == "completed"
    deleted = await conversation.messages(identity.owner_id, chat.id)
    assert all(m.deleted for m in deleted)
    assert not any("远山公司" in m.content or "北岸工作室" in m.content for m in deleted)
    events = await conversation.events(identity.owner_id, turn.id)
    assert "远山公司" not in json.dumps(
        [e.model_dump(mode="json") for e in events], ensure_ascii=False
    )


async def test_current_mention_still_requires_actual_input_span_and_revision(conversation):
    identity, _, turn, source, hint = await current_turn(conversation)
    lease = await conversation.claim("current-scope", model_mode="deepseek")
    await conversation.generation_request(lease)
    async with conversation.db.sessions.begin() as session:
        stored = await session.get(Turn, turn.id)
        actual, reason = await checked_mentions(session, stored, "远山公司。", (hint,))
        assert actual == [(source.id, source.revision)] and reason == "ok"
        context = stored.memory_context
        stored.memory_context = {
            **context,
            "current_source": {
                **context["current_source"],
                "end_offset": 4,
            },
        }
        actual, reason = await checked_mentions(session, stored, "远山公司。", (hint,))
        assert actual is None and reason == "source_permission"
        stored.memory_context = {
            **context,
            "current_source": {
                **context["current_source"],
                "revision": source.revision + 1,
            },
        }
        actual, reason = await checked_mentions(session, stored, "远山公司。", (hint,))
        assert actual is None and reason == "source_permission"
