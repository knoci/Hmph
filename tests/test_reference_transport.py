import json

import httpx
import pytest
from pydantic import SecretStr


def test_reference_calls_keep_real_key_out_of_native_config_and_meter_each_attempt():
    from hmph.reference_transport import ReferenceCallError, ReferenceTransport

    seen = []

    def endpoint(request):
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer test-private-key-not-real"
        body = json.loads(request.content)
        assert body["max_tokens"] == 1024 and body["thinking"] == {"type": "disabled"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
            },
        )

    transport = ReferenceTransport(
        SecretStr("test-private-key-not-real"), max_calls=1, delegate=httpx.MockTransport(endpoint)
    )
    with httpx.Client(transport=transport) as client:
        response = client.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": "Bearer harmless-native-placeholder"},
            json={"model": "deepseek-flash", "messages": [], "max_tokens": 4096},
        )
        assert response.status_code == 200
        with pytest.raises(ReferenceCallError, match="reference_call_budget"):
            client.post(
                "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
            )
        with pytest.raises(ReferenceCallError, match="reference_destination_denied"):
            client.post("https://api.openai.com/chat/completions", json={"model": "deepseek-flash"})
    assert len(seen) == 1
    assert transport.records[0]["usage"]["total_tokens"] == 105
    assert "test-private-key-not-real" not in json.dumps(transport.records)


@pytest.mark.parametrize("status,choices", [(401, None), (200, []), (200, None)])
def test_reference_provider_failures_are_sanitized_and_known_usage_survives(status, choices):
    from hmph.reference_transport import ReferenceCallError, ReferenceTransport

    payload = {
        "error": "provider echoed test-private-key-not-real",
        "choices": choices,
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
    }
    transport = ReferenceTransport(
        SecretStr("test-private-key-not-real"),
        delegate=httpx.MockTransport(lambda _: httpx.Response(status, json=payload)),
    )
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ReferenceCallError) as error:
            client.post(
                "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
            )
    assert "test-private-key-not-real" not in str(error.value)
    assert "test-private-key-not-real" not in json.dumps(transport.records)
    assert transport.records[0]["status"] == "failed"
    if status == 200:
        assert transport.records[0]["usage"]["total_tokens"] == 105
