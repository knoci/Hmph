import json

import httpx
import pytest
from pydantic import SecretStr


async def test_deepseek_stream_hides_reasoning_and_tracks_usage():
    from hmph.deepseek import DeepSeekModel
    from hmph.model_gateway import ChatMessage, GenerationRequest

    seen = []

    def endpoint(request):
        seen.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-key-not-real"
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "PRIVATE_REASONING"}}]},
            {"choices": [{"delta": {"content": "你好。"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            },
        ]
        body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        return httpx.Response(200, text=body + "data: [DONE]\n\n")

    model = DeepSeekModel(SecretStr("test-key-not-real"), transport=httpx.MockTransport(endpoint))
    request = GenerationRequest(messages=(ChatMessage(role="user", content="你好"),))
    deltas = [delta async for delta in model.stream(request)]
    assert "".join(delta.text for delta in deltas) == "你好。"
    assert "PRIVATE_REASONING" not in str(deltas)
    assert deltas[-1].usage.total_tokens == 16
    assert seen[0]["thinking"] == {"type": "disabled"}
    assert seen[0]["stream"] is True
    assert seen[0]["max_tokens"] <= 512


@pytest.mark.parametrize(
    "status,code",
    [(401, "provider_authentication"), (402, "provider_balance"), (429, "provider_rate_limit")],
)
async def test_provider_errors_never_include_key_or_response_body(status, code):
    from hmph.deepseek import DeepSeekModel, ProviderError
    from hmph.model_gateway import ChatMessage, GenerationRequest

    def endpoint(request):
        return httpx.Response(status, text="test-key-not-real PRIVATE_ERROR_BODY")

    model = DeepSeekModel(SecretStr("test-key-not-real"), transport=httpx.MockTransport(endpoint))
    with pytest.raises(ProviderError) as caught:
        async for _ in model.stream(
            GenerationRequest(messages=(ChatMessage(role="user", content="你好"),)),
        ):
            pass
    assert str(caught.value) == code
    assert "test-key-not-real" not in str(caught.value)
    assert "PRIVATE_ERROR_BODY" not in str(caught.value)


async def test_deepseek_worker_reviews_whole_reply_and_retains_usage(conversation):
    from hmph.deepseek import DeepSeekModel
    from hmph.model_gateway import FakeModel
    from hmph.worker import Worker

    def endpoint(request):
        body = json.loads(request.content)
        if body["stream"]:
            chunks = [
                {"choices": [{"delta": {"content": "你好，阿黎。"}}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                },
            ]
            return httpx.Response(
                200,
                text="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                + "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"allow":true,"reason":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 8, "total_tokens": 58},
            },
        )

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(
        identity.owner_id, chat.id, "turn", "我叫阿黎。", model_mode="deepseek"
    )
    assert not await Worker(conversation, FakeModel()).run_once()
    model = DeepSeekModel(SecretStr("test-key-not-real"), transport=httpx.MockTransport(endpoint))
    await Worker(
        conversation,
        model,
        model_mode="deepseek",
        reviewer=model.review,
        model_name="deepseek-flash",
    ).run_once()
    saved = await conversation.get_turn(identity.owner_id, turn.id)
    messages = await conversation.messages(identity.owner_id, chat.id)
    assert saved.status == "completed"
    assert messages[-1].content == "你好，阿黎。"
    assert saved.usage["generation"]["total_tokens"] == 16
    assert saved.usage["guard"]["total_tokens"] == 58
    assert saved.usage["guard_action"] == "allowed"


async def test_generation_request_keeps_current_chat_but_not_other_chat(conversation):
    from hmph.model_gateway import FakeModel
    from hmph.worker import Worker

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat-1")
    other = await conversation.create_chat(identity.owner_id, identity.character_id, "chat-2")
    await conversation.accept(identity.owner_id, other.id, "other", "OTHER_CHAT_SECRET")
    await Worker(conversation, FakeModel()).run_once()
    await conversation.accept(identity.owner_id, chat.id, "first", "我叫阿黎。")
    await Worker(conversation, FakeModel()).run_once()
    await conversation.accept(identity.owner_id, chat.id, "next", "我叫什么？")
    lease = await conversation.claim("test-worker")
    request = await conversation.generation_request(lease)
    assert [message.role for message in request.messages] == ["user", "assistant", "user"]
    assert request.messages[0].content == "我叫阿黎。"
    assert request.messages[-1].content == "我叫什么？"
    assert "OTHER_CHAT_SECRET" not in request.model_dump_json()


@pytest.mark.parametrize("ending", ["", "length"])
async def test_incomplete_or_truncated_stream_is_not_reported_as_complete(ending):
    from hmph.deepseek import DeepSeekModel, ProviderError
    from hmph.model_gateway import ChatMessage, GenerationRequest

    def endpoint(request):
        event = {
            "choices": [{"delta": {"content": "未完成的回复"}, "finish_reason": ending or None}]
        }
        return httpx.Response(200, text="data: " + json.dumps(event) + "\n\n")

    model = DeepSeekModel(SecretStr("test-key-not-real"), transport=httpx.MockTransport(endpoint))
    with pytest.raises(ProviderError, match="provider_stream_incomplete|provider_output_truncated"):
        async for _ in model.stream(
            GenerationRequest(messages=(ChatMessage(role="user", content="好"),))
        ):
            pass


async def test_redirect_is_not_followed_with_credentials():
    from hmph.deepseek import DeepSeekModel, ProviderError
    from hmph.model_gateway import ChatMessage, GenerationRequest

    destinations = []

    def endpoint(request):
        destinations.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://other.invalid/collect"})

    model = DeepSeekModel(SecretStr("test-key-not-real"), transport=httpx.MockTransport(endpoint))
    with pytest.raises(ProviderError, match="provider_request_rejected"):
        async for _ in model.stream(
            GenerationRequest(messages=(ChatMessage(role="user", content="好"),))
        ):
            pass
    assert destinations == ["https://api.deepseek.com/chat/completions"]


async def test_closing_generation_closes_the_provider_stream():
    import asyncio

    from hmph.deepseek import DeepSeekModel
    from hmph.model_gateway import ChatMessage, GenerationRequest

    closed = False

    class ProviderStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            await asyncio.Event().wait()

        async def aclose(self):
            nonlocal closed
            closed = True

    model = DeepSeekModel(
        SecretStr("test-key-not-real"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=ProviderStream()),
        ),
    )
    stream = model.stream(GenerationRequest(messages=(ChatMessage(role="user", content="好"),)))
    assert (await anext(stream)).text == "hello"
    await stream.aclose()
    assert closed


async def test_blocked_real_reply_is_never_published(conversation):
    from hmph.model_gateway import GuardReview, ModelDelta
    from hmph.worker import Worker

    class Provider:
        async def stream(self, request):
            yield ModelDelta(text="UNAPPROVED_DRAFT_SENTINEL")

    async def reject(request, candidate):
        return GuardReview(allowed=False, reason="attack")

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(
        identity.owner_id, chat.id, "turn", "你好", model_mode="deepseek"
    )
    await Worker(conversation, Provider(), model_mode="deepseek", reviewer=reject).run_once()
    messages = await conversation.messages(identity.owner_id, chat.id)
    assert messages[-1].content == "这条回复暂时无法提供，请稍后再试。"
    assert "UNAPPROVED_DRAFT" not in str(await conversation.events(identity.owner_id, turn.id))


async def test_guard_service_failure_has_technical_failure_not_a_user_violation(conversation):
    from hmph.deepseek import ProviderError
    from hmph.model_gateway import ModelDelta
    from hmph.worker import Worker

    class Provider:
        async def stream(self, request):
            yield ModelDelta(text="PRIVATE_UNREVIEWED_SENTINEL")

    async def unavailable(request, candidate):
        raise ProviderError("provider_timeout")

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(
        identity.owner_id, chat.id, "turn", "你好", model_mode="deepseek"
    )
    await Worker(conversation, Provider(), model_mode="deepseek", reviewer=unavailable).run_once()
    saved = await conversation.get_turn(identity.owner_id, turn.id)
    assert (saved.status, saved.error_code) == ("failed", "guard_unavailable")
    assert saved.usage["guard_action"] == "unavailable_fallback"
    assert saved.usage["guard"] is None
    assert "PRIVATE_UNREVIEWED" not in str(await conversation.events(identity.owner_id, turn.id))


async def test_deepseek_extracts_structured_proposals_with_bounded_json_request():
    from hmph.deepseek import DeepSeekModel

    seen = []

    def endpoint(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "proposals": [
                                        {"predicate": "name", "value": "小沐", "quote": "我叫小沐"}
                                    ],
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            },
        )

    model = DeepSeekModel(SecretStr("test-key-not-real"), transport=httpx.MockTransport(endpoint))
    proposals, usage = await model.extract("我叫小沐。")
    assert proposals[0].value == "小沐" and usage.total_tokens == 140
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert seen[0]["thinking"] == {"type": "disabled"}
    assert seen[0]["max_tokens"] <= 512
    assert "test-key-not-real" not in json.dumps(seen)


async def test_state_model_uses_only_structured_state_and_approved_source_messages(conversation):
    from hmph.deepseek import DeepSeekModel
    from hmph.dialogue import Dialogue
    from hmph.dialogue_worker import StateWorker

    seen = []

    def endpoint(request):
        body = json.loads(request.content)
        seen.append(body)
        source = json.loads(body["messages"][-1]["content"])["messages"][0]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "goal": {
                                        "message_id": source["id"],
                                        "text": "周五面试",
                                        "quote": "周五面试",
                                    }
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            },
        )

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    turn = await conversation.accept(
        identity.owner_id, chat.id, "plan", "周五面试。", model_mode="deepseek"
    )
    await conversation.cancel(identity.owner_id, turn.id, "cancel")
    model = DeepSeekModel(SecretStr("test-key-not-real"), transport=httpx.MockTransport(endpoint))
    dialogue = Dialogue(conversation.db)
    assert await StateWorker(dialogue, model.update_state, model_mode="deepseek").run_once()
    assert (await dialogue.read(identity.owner_id, chat.id)).data.goal.text == "周五面试"
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert "test-key-not-real" not in json.dumps(seen)
