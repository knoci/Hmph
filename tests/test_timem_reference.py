from types import SimpleNamespace
from uuid import uuid4

import pytest

from hmph.eval_budget import RunBudget
from hmph.timem_reference import TiMemSupplement, leaf_sources


def test_native_leaf_closure_is_complete_and_deduplicated():
    nodes = {
        "day": SimpleNamespace(child_memory_ids=["session", "leaf2"]),
        "session": SimpleNamespace(child_memory_ids=["leaf1", "leaf2"]),
    }
    assert leaf_sources("day", nodes, {"leaf1": ["m1"], "leaf2": ["m1", "m2"]}) == ("m1", "m2")


@pytest.mark.parametrize(
    "nodes",
    [
        {"day": SimpleNamespace(child_memory_ids=["missing"])},
        {"day": SimpleNamespace(child_memory_ids=["day"])},
    ],
)
def test_native_unknown_or_cyclic_provenance_is_not_silently_dropped(nodes):
    with pytest.raises(ValueError):
        leaf_sources("day", nodes, {})


def test_timem_has_distinct_persistent_budget_identity(tmp_path):
    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=1, max_input_bytes=100)
    binding = budget.bind("case-001", "timem", "writer")
    reservation = binding.acquire(50)
    binding.finish(reservation, {"status": "completed", "usage": None})
    row = budget.snapshot()["calls"][0]
    assert row["profile"] == "timem"
    assert row["record"]["usage"] is None


async def test_original_recall_uses_event_date_even_when_recorded_later():
    from datetime import UTC, datetime

    from hmph.memory_build_types import BuildSource
    from hmph.timem_reference import LocalTiMem

    late, recent = uuid4(), uuid4()
    observed = datetime(2026, 2, 8, 18, tzinfo=UTC)
    sources = tuple(
        BuildSource(
            message_id=identity, revision=1, source_order=order, content=content,
            observed_at=observed, source_timezone="Asia/Hong_Kong",
        )
        for order, (identity, content) in enumerate((
            (recent, "蓝鲸工作室聚餐今天去了梧桐饭馆。"),
            (late, "蓝鲸工作室聚餐昨天去了白桦饭馆。"),
        ), 1)
    )
    native = LocalTiMem()
    native.snapshot = sources
    native.embedding = SimpleNamespace(embed=lambda _: iter([SimpleNamespace(tolist=lambda: [])]))
    native.client = SimpleNamespace(
        collection_exists=lambda _: True,
        query_points=lambda *_, **__: SimpleNamespace(points=[SimpleNamespace(id=str(recent))]),
    )
    candidates = await native.original_sources(
        "2026年2月8日蓝鲸工作室聚餐去了哪？",
        reference_time=datetime(2026, 2, 10, 12, tzinfo=UTC),
        source_timezone="Asia/Hong_Kong",
    )
    assert candidates[0] == str(late)


async def test_original_recall_keeps_a_linked_correction_with_a_different_surface_name():
    from datetime import UTC, datetime

    from hmph.memory_build_types import BuildSource
    from hmph.timem_reference import LocalTiMem

    original, irrelevant, corrected = uuid4(), uuid4(), uuid4()
    messages = (
        (original, "桥灯合唱团的新曲暂叫《岸边夜色》，周末试演。"),
        (irrelevant, "更正一下，冰箱换成三门型，我找人搬过来了。"),
        (corrected, "周末试演那首，曲名不是《岸边夜色》，是《雨后的桥》。"),
    )
    native = LocalTiMem()
    native.snapshot = tuple(
        BuildSource(message_id=identity, revision=1, source_order=order, content=content,
                    observed_at=datetime(2026, 2, 8, tzinfo=UTC), source_timezone="UTC")
        for order, (identity, content) in enumerate(messages, 1)
    )
    native.embedding = SimpleNamespace(embed=lambda _: iter([SimpleNamespace(tolist=lambda: [])]))
    native.client = SimpleNamespace(
        collection_exists=lambda _: True,
        query_points=lambda *_, **__: SimpleNamespace(points=[SimpleNamespace(id=str(original))]),
    )
    sources = await native.original_sources("桥灯合唱团的新曲最终叫什么？")
    assert sources[0] == str(corrected)
    assert str(irrelevant) not in sources[:2]


async def test_original_recall_ranks_adjacent_condition_with_its_withdrawal():
    from datetime import UTC, datetime

    from hmph.memory_build_types import BuildSource
    from hmph.timem_reference import LocalTiMem

    original, followup, distraction, unrelated = (uuid4() for _ in range(4))
    chat, other, elsewhere = uuid4(), uuid4(), uuid4()
    native = LocalTiMem()
    native.snapshot = tuple(
        BuildSource(message_id=identity, revision=1, source_order=order, content=content,
                    observed_at=datetime(2026, 2, 8, tzinfo=UTC), source_timezone="UTC",
                    conversation_id=chat_id)
        for order, (identity, chat_id, content) in enumerate((
            (original, chat, "晨间天气提醒已经撤掉了。"),
            (followup, chat, "以后除非我主动说，才需要问。"),
            (distraction, other, "晨间天气提醒也已经撤掉了。"),
            (unrelated, elsewhere, "除非我主动说，否则不用问早餐。"),
        ), 1)
    )
    native.embedding = SimpleNamespace(embed=lambda _: iter([SimpleNamespace(tolist=lambda: [])]))
    native.client = SimpleNamespace(
        collection_exists=lambda _: True,
        query_points=lambda *_, **__: SimpleNamespace(points=[SimpleNamespace(id=str(original))]),
    )
    candidates = await native.original_sources("现在还要主动提晨间天气提醒吗？")
    assert candidates[0] == str(followup)
    assert str(unrelated) not in candidates


async def test_combined_build_checks_the_lease_limit_before_starting_native_writes(monkeypatch):
    from hmph.mem0_backend import Mem0Backend
    from hmph.memory_build_types import FrameworkBuildError
    from hmph.timem_runtime import BACKEND, REVISION, CombinedBackend

    async def must_not_write(*args):
        pytest.fail("Native writer started even though the lease could not fund the build")

    monkeypatch.setattr(Mem0Backend, "build", must_not_write)
    backend = object.__new__(CombinedBackend)
    backend.batch_bytes, backend.max_calls = 3000, 12
    lease = SimpleNamespace(
        backend=BACKEND,
        backend_revision=REVISION,
        call_limit=2,
        reused_sources=0,
        sources=(
            SimpleNamespace(content="短消息一。", conversation_id=1),
            SimpleNamespace(content="短消息二。", conversation_id=2),
        ),
    )
    with pytest.raises(FrameworkBuildError, match="call_budget"):
        await backend.build(lease, must_not_write)


async def test_timed_out_ranking_keeps_completed_hierarchical_evidence():
    from hmph.online_memory import NativeSources

    ticket = SimpleNamespace(build_id=uuid4(), space_id=uuid4())
    fact, episode = uuid4(), uuid4()

    class Mem0:
        async def retrieve_sources(self, ticket, query):
            return NativeSources(source_ids=(fact,))

    class TiMem:
        async def retrieve(self, query):
            return (), (str(episode),)

        async def episode_sources(self, query):
            raise TimeoutError

    reader = TiMemSupplement(
        Mem0(), TiMem(), build_id=ticket.build_id, space_id=ticket.space_id, originals=True
    )
    result = await reader.retrieve_sources(ticket, "记得上次发生了什么？")
    assert result.source_ids == (fact, episode)
    assert result.timem_sources == (episode,)
    assert result.branch_status == {"mem0": "ok", "timem": "partial_timeout"}


async def test_combined_candidates_keep_both_current_fact_and_episode_under_budget():
    from hmph.online_memory import NativeSources

    ticket = SimpleNamespace(build_id=uuid4(), space_id=uuid4(), reference_time=None)
    originals = tuple(uuid4() for _ in range(8))
    old, correction, linked_event = uuid4(), uuid4(), uuid4()

    class Mem0:
        async def retrieve_sources(self, ticket, query):
            return NativeSources(source_ids=(old, correction))

    class TiMem:
        snapshot = ()

        async def retrieve(self, query):
            return (), (str(linked_event),)

        async def episode_sources(self, query):
            return ()

        async def original_sources(self, query):
            return tuple(map(str, originals))

    reader = TiMemSupplement(
        Mem0(), TiMem(), build_id=ticket.build_id, space_id=ticket.space_id, originals=True
    )
    result = await reader.retrieve_sources(ticket, "这件事后来改成什么了？")
    assert {originals[0], old, correction, linked_event} <= set(result.source_ids[:6])
    assert originals[-1] not in result.source_ids


async def test_current_withdrawal_keeps_a_candidate_buffer_for_scoped_packing():
    from hmph.online_memory import NativeSources

    ticket = SimpleNamespace(build_id=uuid4(), space_id=uuid4(), reference_time=None)
    change, support = uuid4(), uuid4()
    other = tuple(uuid4() for _ in range(10))

    class Mem0:
        async def retrieve_sources(self, ticket, query):
            return NativeSources(source_ids=(*other, support))

    class TiMem:
        snapshot = (SimpleNamespace(
            content="晚间喝水提醒撤掉。", message_id=change, source_order=1, conversation_id=None
        ),)

        async def retrieve(self, query):
            return (), ()

        async def episode_sources(self, query):
            return ()

        async def original_sources(self, query):
            return (str(change),)

    reader = TiMemSupplement(
        Mem0(), TiMem(), build_id=ticket.build_id, space_id=ticket.space_id, originals=True
    )
    result = await reader.retrieve_sources(ticket, "现在还要晚间喝水提醒吗？")
    assert result.source_ids[0] == change
    assert support in result.source_ids  # Overfetch is only here, before the common context cap.
    ordinary = await reader.retrieve_sources(ticket, "晚间聊过什么？")
    assert support not in ordinary.source_ids


async def test_empty_combination_keeps_the_branch_timeout_instead_of_reporting_no_match():
    import time

    from hmph.online_memory import retrieve_native

    ticket = SimpleNamespace(build_id=uuid4(), space_id=uuid4())

    class Mem0:
        async def retrieve_sources(self, ticket, query):
            raise TimeoutError

    class TiMem:
        async def retrieve(self, query):
            return (), ()

    reader = TiMemSupplement(Mem0(), TiMem(), build_id=ticket.build_id, space_id=ticket.space_id)
    result, status = await retrieve_native(
        reader, ticket, "过去的事", deadline=time.monotonic() + 4
    )
    assert result.source_ids == ()
    assert result.branch_status == {"mem0": "timeout", "timem": "empty"}
    assert status == "timeout"


async def test_native_writer_provider_is_bound_to_its_execution_context(monkeypatch):
    import asyncio
    import sys
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    from hmph.timem_reference import install_provider

    fake = ModuleType("llm")
    fake.__spec__ = ModuleSpec("llm", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "llm", fake)
    monkeypatch.setitem(sys.modules, "llm.llm_manager", ModuleType("llm.llm_manager"))
    first, second = object(), object()
    install_provider(first)
    imported = sys.modules["llm.llm_manager"].get_llm

    async def another_build():
        install_provider(second)
        return imported()

    assert await asyncio.create_task(another_build()) is second
    assert imported() is first


async def test_provider_failure_latches_before_a_second_request(tmp_path):
    pytest.importorskip("openai", reason="Provider test runs in isolated TiMem environment")
    from pydantic import SecretStr

    from hmph.reference_transport import ReferenceCallError
    from hmph.timem_reference import MeteredTiMemLLM

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=2, max_input_bytes=100)
    llm = MeteredTiMemLLM(SecretStr("test-only-placeholder"), budget, "case-001")
    calls = []

    async def fail(**kwargs):
        calls.append(1)
        raise RuntimeError("fixture failure")

    llm.sdk.chat.completions.create = fail
    try:
        with pytest.raises(RuntimeError):
            await llm.chat([])
        with pytest.raises(ReferenceCallError):
            await llm.chat([])
        assert calls == [1]
    finally:
        await llm.close()
