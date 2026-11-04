import asyncio

import pytest

from hmph.contracts import DomainError
from hmph.memory import Memory
from hmph.model_gateway import ModelUsage
from hmph.recall import RecallDecision
from hmph.working_memory import FocusPlan


async def bridge_history(conversation):
    identity = await conversation.bootstrap()
    history = await conversation.create_chat(
        identity.owner_id, identity.character_id, "recall-history"
    )
    sources = []
    for index, content in enumerate(("我姐姐叫露西。", "露西在里昂的一家银行上班。")):
        turn = await conversation.accept(identity.owner_id, history.id, str(index), content)
        lease = await conversation.claim("recall-history")
        await conversation.finish(lease)
        sources.append(
            next(
                m
                for m in await conversation.messages(identity.owner_id, history.id)
                if m.turn_id == turn.id
            )
        )
    current = await conversation.create_chat(
        identity.owner_id, identity.character_id, "recall-query"
    )
    turn = await conversation.accept(
        identity.owner_id, current.id, "query", "我姐姐在哪个城市工作？"
    )
    lease = await conversation.claim("recall-query")
    return identity, sources, turn, lease


def bridge_decision(source_id):
    return RecallDecision(
        plan=FocusPlan.model_validate(
            {
                "bridge": {
                    "source_id": str(source_id),
                    "quote": "露西",
                    "slot": "entity",
                }
            }
        ),
        usage=ModelUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )


async def test_product_bridge_retrieves_second_hop_and_accounts_for_one_call(conversation):
    identity, sources, turn, lease = await bridge_history(conversation)
    calls = []

    async def planner(request):
        calls.append(request)
        assert str(sources[0].id) in {s.source_id for s in request.sources}
        assert str(sources[1].id) not in {s.source_id for s in request.sources}
        return bridge_decision(sources[0].id)

    conversation.recall_planner = planner
    result = await conversation.generation_request(lease)
    assert len(calls) == 1
    assert {s.id for s in sources} <= {s.message_id for s in result.raw_evidence}
    assert result.workspace.candidate_count <= 80
    assert result.workspace.rounds == 2 and result.workspace.query_count <= 4
    assert result.workspace.query_hint_source_ids == (sources[0].id,)
    explanation = await conversation.memory_explanation(identity.owner_id, turn.id)
    assert explanation["context"]["planner"]["status"] == "applied"
    finished = await conversation.finish(lease)
    assert finished.usage["planner"]["usage"]["total_tokens"] == 120


async def test_unknown_bridge_cannot_retrieve_or_claim_new_support(conversation):
    identity, sources, turn, lease = await bridge_history(conversation)

    async def planner(request):
        return bridge_decision(sources[1].id)  # A real source, absent from the supplied snippets.

    conversation.recall_planner = planner
    result = await conversation.generation_request(lease)
    assert sources[1].id not in {s.message_id for s in result.raw_evidence}
    explanation = await conversation.memory_explanation(identity.owner_id, turn.id)
    assert explanation["context"]["planner"]["status"] == "invalid"


async def test_planner_timeout_is_bounded_and_not_retried(conversation, monkeypatch):
    import hmph.conversation as module

    identity, _, turn, lease = await bridge_history(conversation)
    monkeypatch.setattr(module, "READ_SECONDS", 0.15)
    calls = []

    async def planner(request):
        calls.append(request)
        await asyncio.Event().wait()

    conversation.recall_planner = planner
    await conversation.generation_request(lease)
    await conversation.generation_request(lease)
    assert len(calls) == 1
    explanation = await conversation.memory_explanation(identity.owner_id, turn.id)
    assert explanation["context"]["planner"]["status"] == "timeout"
    finished = await conversation.finish(lease)
    assert finished.usage["complete"] is False
    assert finished.usage["planner"]["usage"] is None


async def test_forget_during_planning_invalidates_the_old_draft(conversation):
    identity, sources, _, lease = await bridge_history(conversation)
    called, release = asyncio.Event(), asyncio.Event()

    async def planner(request):
        called.set()
        await release.wait()
        return bridge_decision(sources[0].id)

    conversation.recall_planner = planner
    task = asyncio.create_task(conversation.generation_request(lease))
    await asyncio.wait_for(called.wait(), timeout=2)
    try:
        memory = Memory(conversation.db, digest_key=conversation.digest_key)
        await memory.forget_source(
            identity.owner_id, sources[0].id, "forget-bridge", expected_revision=sources[0].revision
        )
    finally:
        release.set()
    with pytest.raises(DomainError, match="stale_lease"):
        await task


async def test_default_off_and_pause_or_coverage_gap_do_not_create_planner_input(conversation):
    from hmph.recall import prepare_recall

    identity, _, turn, lease = await bridge_history(conversation)
    baseline = await conversation.generation_request(lease)
    assert conversation.recall_planner is None
    assert prepare_recall(baseline) is not None
    explanation = await conversation.memory_explanation(identity.owner_id, turn.id)
    assert "planner" not in explanation["context"]
    for change in ({"shared_thread_action": "pause"}, {"thread_control": "paused"}):
        paused = baseline.model_copy(
            update={
                "companion": baseline.companion.model_copy(update=change),
            }
        )
        assert prepare_recall(paused) is None
    for flag in ("pending_gap", "state_truncated"):
        incomplete = baseline.model_copy(
            update={
                "short_term": baseline.short_term.model_copy(update={flag: True}),
            }
        )
        assert prepare_recall(incomplete) is None
    withheld = baseline.model_copy(
        update={
            "companion": baseline.companion.model_copy(
                update={
                    "memory_use": tuple(
                        d.model_copy(update={"permission": "omit"})
                        for d in baseline.companion.memory_use
                    ),
                }
            )
        }
    )
    assert prepare_recall(withheld) is None
