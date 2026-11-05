from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from hmph.eval_budget import RunBudget
from hmph.reference_transport import AsyncReferenceTransport, ReferenceCallError, ReferenceTransport


@pytest.fixture
def public_helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from public_memory import pack
    from selection_profile import transport

    return pack, transport


def test_public_episode_packing_preserves_long_messages_and_keeps_gold_out(public_helpers):
    pack, _ = public_helpers
    content = "甲乙丙丁" * 1000
    record = {
        "dataset": "fixture",
        "timestamp": "2023/01/02 10:00",
        "question": "发生了什么？",
        "answer": "SCORING_ONLY_SECRET",
        "sources": ["original-1"],
        "history": [
            {
                "source_id": "original-1",
                "session_id": "session-1",
                "timestamp": "2023/01/01 12:00",
                "role": "assistant",
                "content": content,
            }
        ],
    }
    case, mapping = pack(record)
    joined = "".join(s.content.replace("\n", "") for s in case.history)
    assert joined == "[original-1] assistant: " + content
    assert all(len(s.content.encode()) <= 3000 for s in case.history)
    assert all(v == ["original-1"] for v in mapping.values())
    assert "SCORING_ONLY_SECRET" not in str(case.writer_input())
    assert all(s.observed_at <= case.query.timestamp for s in case.history)


def test_public_native_stop_reaches_reused_transport_before_any_reservation(
    tmp_path, public_helpers
):
    _, transport = public_helpers
    stop = tmp_path / "STOP"
    stop.touch()
    budget = RunBudget(
        tmp_path / "calls.sqlite", max_calls=4, max_input_bytes=16384, per_case_calls=4
    )
    request = httpx.Request(
        "POST", "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
    )
    for profile in ("mem0", "graphiti"):
        writer = transport(
            SecretStr("not-a-real-key"),
            "deepseek-flash",
            budget,
            "case-001",
            profile,
            "writer",
            stop_file=stop,
        )
        with pytest.raises(ReferenceCallError, match="reference_cancelled"):
            writer.handle_request(request)
        budget.bind("case-001", profile, "judge")
    assert budget.snapshot()["calls"] == []


@pytest.mark.parametrize("echo_credential", [False, True])
def test_failed_native_response_can_be_captured_without_headers_or_credentials(echo_credential):
    captured = []
    credential = "test-only-credential-sentinel"
    delegate = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": credential
                            if echo_credential
                            else "truncated structured output"
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )
    )
    writer = ReferenceTransport(
        SecretStr(credential), delegate=delegate, failure_observer=captured.append
    )
    request = httpx.Request(
        "POST",
        "https://api.deepseek.com/chat/completions",
        json={
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "public fixture"}],
            "max_tokens": 2048,
        },
    )
    with pytest.raises(ReferenceCallError, match="reference_incomplete_output"):
        writer.handle_request(request)
    assert writer.records[0]["usage"]["total_tokens"] == 30
    if echo_credential:
        assert not captured and writer.records[0]["failure_capture"] == "credential_redacted"
    else:
        assert captured[0]["request"]["max_tokens"] == 1024
        assert "Authorization" not in str(captured)
        assert credential not in str(captured)


async def test_offline_network_retry_is_bounded_charged_and_excludes_truncation(tmp_path):
    attempts = 0

    def transient(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.RemoteProtocolError("synthetic disconnect")
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    budget = RunBudget(
        tmp_path / "calls.sqlite", max_calls=3, max_input_bytes=10000, per_case_calls=3
    )
    writer = AsyncReferenceTransport(
        SecretStr("fixture"),
        delegate=httpx.MockTransport(transient),
        network_retries=1,
        budget=budget.bind("case-001", "graphiti", "writer"),
    )
    request = httpx.Request(
        "POST", "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
    )
    response = await writer.handle_async_request(request)
    assert response.status_code == 200 and attempts == 2
    records = budget.snapshot()["calls"]
    assert len(records) == 2 and records[0]["record"]["usage"] is None
    assert writer.records[0]["exception_type"] == "RemoteProtocolError"
    await writer.aclose()
    attempts = 0

    def truncated(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"choices": [{"finish_reason": "length"}]})

    writer = AsyncReferenceTransport(
        SecretStr("fixture"), delegate=httpx.MockTransport(truncated), network_retries=1
    )
    with pytest.raises(ReferenceCallError, match="reference_incomplete_output"):
        await writer.handle_async_request(request)
    assert attempts == 1
    await writer.aclose()


async def test_public_product_baseline_shares_recent_turns_without_memory(public_helpers):
    import json
    from datetime import UTC, datetime
    from uuid import uuid4

    from public_product import paired_memory_requests

    from hmph.companion_policy import CompanionPlan
    from hmph.deepseek import DeepSeekModel
    from hmph.dialogue_types import ShortTermContext
    from hmph.model_gateway import ChatMessage, ContextOrigin, GenerationRequest
    from hmph.preference_types import ExpressionPreferences
    from hmph.working_memory import RawEvidence

    messages = tuple(
        ChatMessage(role=role, content=f"{role}-{turn}")
        for turn in range(6)
        for role in ("user", "assistant")
    ) + (ChatMessage(role="user", content="current"),)
    origins = tuple(
        ContextOrigin(
            message_id=uuid4(),
            message_revision=1,
            start_offset=0,
            end_offset=len(message.content),
        )
        for message in messages
    )
    observed = datetime(2023, 3, 15, tzinfo=UTC)
    memory = RawEvidence(
        message_id=uuid4(), message_revision=1, role="user", recorded_at=observed,
        content="window-external-fact", start_offset=0, end_offset=20, truncated=False,
    )
    request = GenerationRequest(
        messages=messages,
        history_origins=origins,
        raw_evidence=(memory,),
        short_term=ShortTermContext(revision=7),
        preferences=ExpressionPreferences(nickname="private-nickname"),
        companion=CompanionPlan(
            policy_revision=7, act="recall", support_mode="unknown", tone="warm",
            operator="none", shared_thread={"topic": "window-external-plan"},
        ),
        reference_time=observed,
    )
    full, recent_only = paired_memory_requests(request)
    assert full.messages == recent_only.messages == messages[-9:]
    assert full.history_origins == recent_only.history_origins == origins[-9:]
    assert full.reference_time == recent_only.reference_time == observed

    calls = []

    def endpoint(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}\n\ndata: [DONE]\n\n',
        )

    model = DeepSeekModel(SecretStr("mock"), transport=httpx.MockTransport(endpoint))
    for candidate in (full, recent_only):
        _ = [delta async for delta in model.stream(candidate)]
    for marker in ("window-external-fact", "window-external-plan", "private-nickname"):
        assert marker in json.dumps(calls[0])
        assert marker not in json.dumps(calls[1])
    assert calls[0]["messages"][-9:] == calls[1]["messages"][-9:]
    assert calls[0]["messages"][-1]["content"] == "current"


async def test_public_product_import_preserves_clock_and_reaches_existing_read_path(
    conversation, public_helpers
):
    from datetime import UTC, datetime
    from uuid import uuid4

    from public_product import import_history
    from sqlalchemy import select

    from hmph.models import Turn
    from hmph.selection_eval import Gold, Query, SelectionCase, Source

    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
    observed = datetime(2023, 3, 15, tzinfo=UTC)
    case = SelectionCase(
        "case-001",
        "fixture",
        (
            Source(
                source_id="s001", observed_at=observed, content="user: I read The New Yorker today."
            ),
        ),
        Query(question="What did I read?", timestamp=observed),
        Gold(
            answer="The New Yorker", aliases=(), evidence_sources=("s001",), rationale="score only"
        ),
    )
    mapping = await import_history(conversation.db, chat.id, case)
    assert len(mapping) == 1
    async with conversation.db.sessions() as session:
        original = await session.scalar(select(Turn).where(Turn.conversation_id == chat.id))
        assert original.created_at == observed
        assert original.usage["generation_calls"] == 0
    await conversation.accept(identity.owner_id, chat.id, uuid4().hex, "我读过哪本杂志？")
    lease = await conversation.claim(uuid4().hex, 30)
    request = await conversation.generation_request(lease)
    assert any("The New Yorker" in m.content for m in request.messages)
