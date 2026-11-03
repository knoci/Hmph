"""Product evaluation uses real input/output seams, not a second short-answer reader."""

import json
from dataclasses import replace

import httpx
import pytest
from pydantic import SecretStr

from hmph.eval_budget import RunBudget


async def test_product_stream_is_metered_without_changing_the_sse_body(tmp_path):
    from hmph.product_eval import ProductTransport

    budget = RunBudget(
        tmp_path / "calls.sqlite", max_calls=2, max_input_bytes=4096, per_case_calls=2
    )
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    stream = "data: " + json.dumps({"choices": [{"delta": {"content": "你好。"}}]}) + "\n\n"
    stream += (
        "data: "
        + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage})
        + "\n\ndata: [DONE]\n\n"
    )

    def endpoint(request):
        assert request.headers["Authorization"] == "Bearer local-test-not-a-real-key"
        return httpx.Response(200, text=stream)

    transport = ProductTransport(
        SecretStr("local-test-not-a-real-key"),
        budget=budget.bind("p2-dev-001", "lexical", "generation"),
        delegate=httpx.MockTransport(endpoint),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        result = await client.post(
            "https://api.deepseek.com/chat/completions",
            json={
                "model": "deepseek-flash",
                "stream": True,
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 256,
            },
        )
    assert result.text == stream
    record = budget.snapshot()["calls"][0]["record"]
    assert record["usage"] == usage and record["status"] == "completed"
    assert "local-test-not-a-real-key" not in json.dumps(budget.snapshot())


async def test_product_context_records_exact_history_origins_and_current_reference_time(
    conversation,
):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    first = await conversation.accept(identity.owner_id, chat.id, "old", "我喜欢跑步。")
    await conversation.cancel(identity.owner_id, first.id, "cancel")
    await conversation.accept(identity.owner_id, chat.id, "query", "记得我喜欢什么运动吗？")
    request = await conversation.generation_request(await conversation.claim("eval"))
    assert request.reference_time.tzinfo is not None
    stored = {m.id: m for m in await conversation.messages(identity.owner_id, chat.id)}
    assert len(request.history_origins) == len(request.messages)
    for message, origin in zip(request.messages, request.history_origins, strict=True):
        assert (
            stored[origin.message_id].content[origin.start_offset : origin.end_offset]
            == message.content
        )


def test_product_scores_keep_available_evidence_separate_from_supported_answer():
    from hmph.product_eval import AnswerScore, GroundingScore, grade_product, smoke_case

    case = smoke_case()
    context = {"evidence": [{"source_ids": ["s001"]}, {"source_ids": ["s002"]}]}
    correct = AnswerScore(answer_correct=True, abstains=False, reason="correct")
    # A correct answer with the wrong actual support must fail the joint gate.
    grounded = GroundingScore(grounded=True, source_ids=["s002"], reason="supported")
    score = grade_product(case, context, grounded, correct)
    assert score["answer_correct"] and score["context_required_evidence_covered"]
    assert not score["answer_evidence_supported"] and not score["correct"]
    grounded = grounded.model_copy(update={"source_ids": ["s001", "invented"]})
    score = grade_product(case, context, grounded, correct)
    assert not score["grounding_judge_source_ids_valid"] and not score["correct"]


def test_product_abstention_requires_no_factual_claim_and_empty_references():
    from hmph.product_eval import AnswerScore, GroundingScore, grade_product, smoke_case

    case = smoke_case()
    case = replace(case, gold=case.gold.model_copy(update={"answer": None, "evidence_sources": ()}))
    context = {"evidence": [{"source_ids": ["s001"]}]}
    answer = AnswerScore(answer_correct=True, abstains=True, reason="correct")
    grounded = GroundingScore(grounded=True, source_ids=[], reason="no_factual_answer")
    assert grade_product(case, context, grounded, answer)["correct"]
    with pytest.raises(ValueError):
        GroundingScore(grounded=True, source_ids=["s001"], reason="no_factual_answer")
    with pytest.raises(ValueError):
        GroundingScore(grounded=True, source_ids=[], reason="unsupported")
    with pytest.raises(ValueError):
        AnswerScore(answer_correct=True, abstains=False, reason="wrong")


def test_product_input_cannot_carry_gold_or_category():
    from hmph.product_eval import ProductInput, smoke_case

    case = smoke_case()
    value = dict(
        history=case.history,
        query=case.query,
        source_metadata=case.source_metadata,
        build_watermark=case.build_watermark,
    )
    assert ProductInput(**value).query_timezone == "UTC"
    with pytest.raises(ValueError):
        ProductInput(**value, gold=case.gold)
    with pytest.raises(ValueError):
        ProductInput(**value, category=case.category)


async def test_product_provider_failure_spends_reservation_and_cannot_retry_over_budget(tmp_path):
    from hmph.product_eval import ProductTransport
    from hmph.reference_transport import ReferenceCallError

    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(402, text="sensitive provider error must not enter ledger")

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=1, max_input_bytes=4096)
    transport = ProductTransport(
        SecretStr("local-test"),
        budget=budget.bind("p2-dev-999", "lexical", "guard"),
        delegate=httpx.MockTransport(endpoint),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.post(
            "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
        )
        assert response.status_code == 402
        with pytest.raises(ReferenceCallError):
            await client.post(
                "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
            )
    assert len(requests) == 1 and transport.exhausted
    snapshot = budget.snapshot()
    assert snapshot["calls"][0]["record"]["http_status"] == 402
    assert "sensitive provider" not in json.dumps(snapshot)
    await transport.shutdown()


def test_grounding_cannot_see_dates_absent_from_actual_history_prompt():
    from uuid import uuid4

    from hmph.model_gateway import ChatMessage, ContextOrigin, GenerationRequest
    from hmph.product_eval import ProductInput, actual_context, smoke_case

    case = smoke_case()
    mid = uuid4()
    content = case.history[0].content
    request = GenerationRequest(
        messages=(ChatMessage(role="user", content=content),),
        history_origins=(
            ContextOrigin(
                message_id=mid, message_revision=1, start_offset=0, end_offset=len(content)
            ),
        ),
        reference_time=case.query.timestamp,
    )
    scenario = ProductInput(
        history=case.history,
        query=case.query,
        source_metadata=case.source_metadata,
        build_watermark=case.build_watermark,
    )
    context = actual_context(request, scenario, {"s001": mid}, {})
    assert context["evidence"] == [
        {"source_ids": ["s001"], "content": content, "channel": "history"}
    ]
    assert str(mid) not in json.dumps(context)
    assert case.history[0].observed_at.isoformat() not in json.dumps(context)
