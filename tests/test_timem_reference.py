from types import SimpleNamespace

import pytest

from hmph.eval_budget import RunBudget
from hmph.timem_reference import leaf_sources


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
