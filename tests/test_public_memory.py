from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from hmph.eval_budget import RunBudget
from hmph.reference_transport import ReferenceCallError, ReferenceTransport


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
