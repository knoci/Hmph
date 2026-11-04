import json

import httpx
import pytest
from pydantic import SecretStr

from hmph.deepseek import DeepSeekModel
from hmph.recall import RecallError, RecallInput, RecallSource


async def test_invalid_planner_payload_preserves_reported_usage():
    def provider(request):
        body = json.loads(request.content)
        assert body["max_tokens"] == 768 and body["stream"] is False
        assert "fixture-provider-key" not in request.content.decode()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"bridge": None, "answer": "untrusted extra fact"}
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    model = DeepSeekModel(
        SecretStr("fixture-provider-key"), transport=httpx.MockTransport(provider)
    )
    with pytest.raises(RecallError) as raised:
        await model.plan_recall(
            RecallInput(question="Where?", missing_slots=("answer",), sources=())
        )
    assert raised.value.status == "invalid"
    assert raised.value.usage.total_tokens == 15


async def test_planner_byte_limit_rejects_before_provider_call():
    def denied(request):
        raise AssertionError("Oversized request must not leave the process")

    model = DeepSeekModel(SecretStr("fixture-provider-key"), transport=httpx.MockTransport(denied))
    with pytest.raises(RecallError, match="input_limit"):
        await model.plan_recall(
            RecallInput(
                question="Where?",
                missing_slots=("answer",),
                sources=(RecallSource(source_id="one", revision=1, content="x" * 9000),),
            )
        )
