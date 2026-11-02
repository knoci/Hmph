"""P1 public native adapter + common reader; only provider HTTPS is replaced."""

import json
from importlib.util import find_spec
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

if find_spec("mem0") is None:
    pytest.skip("Run in the framework environment", allow_module_level=True)


def test_selection_native_write_source_expansion_reader_and_cleanup(tmp_path, monkeypatch):
    from hmph.eval_budget import RunBudget
    from hmph.selection_eval import grade, load_selection
    from scripts.selection_profile import mem0_context, read_answer

    cases, _ = load_selection(Path("evals/fixtures/framework-selection-dev.json"))
    case = cases[0]
    seen = []

    def endpoint(request):
        body = json.loads(request.content)
        seen.append(body)
        is_reader = '"answer"' in body["messages"][0]["content"]
        if is_reader:
            content = {"answer": "Lumen Works", "source_ids": ["s002"]}
        else:
            assert case.query.question not in json.dumps(body)
            company = "Lumen Works" if "Lumen Works" in json.dumps(body) else "Alder Labs"
            content = {"memory": [{"text": f"User works at {company}."}]}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    monkeypatch.setattr(httpx, "HTTPTransport", lambda **_: httpx.MockTransport(endpoint))
    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=10, max_input_bytes=100000)
    cleanup, timings = {}, {}
    key = SecretStr("fake-native-selection-key")
    row = {"timings": timings}
    with mem0_context(
        case.history,
        case.query,
        key,
        "deepseek-flash",
        budget,
        case.case_id,
        cleanup,
        None,
        timings,
    ) as evidence:
        assert {sid for item in evidence for sid in item["source_ids"]} == {"s001", "s002"}
        reply, ids, _ = read_answer(
            case.query, evidence, key, "deepseek-flash", budget, case.case_id, "mem0", row
        )
        assert grade(case, reply, ids)["correct"]
    assert cleanup == {"schema_removed": True, "sqlite_history_removed": True}
    assert len(budget.snapshot()["calls"]) == 3
    for body in seen:
        assert "rationale" not in json.dumps(body) and "case-001" not in json.dumps(body)
        assert "fake-native-selection-key" not in json.dumps(body)
