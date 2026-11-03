import json

import httpx
import pytest
from pydantic import SecretStr


@pytest.mark.parametrize(
    "failure,code,phase",
    [
        ("timeout", "reference_timeout", "response_body"),
        ("network", "reference_network", "request"),
        ("503", "reference_service_unavailable", "response_body"),
        ("protocol", "reference_protocol", "response_protocol"),
    ],
)
@pytest.mark.parametrize("kind", ["native", "product"])
async def test_failure_location_is_fixed_and_sanitized(tmp_path, failure, code, phase, kind):
    from hmph.eval_budget import RunBudget
    from hmph.product_eval import ProductTransport
    from hmph.reference_transport import ReferenceCallError, ReferenceTransport

    class PartialSync(httpx.SyncByteStream):
        def __iter__(self):
            yield b"{"
            raise httpx.ReadTimeout("PRIVATE provider body and credential")

    class PartialAsync(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"{"
            raise httpx.ReadTimeout("PRIVATE provider body and credential")

    def endpoint(request):
        if failure == "network":
            raise httpx.ConnectError("PRIVATE provider body and credential")
        if failure == "timeout":
            return httpx.Response(200, stream=PartialSync() if kind == "native" else PartialAsync())
        return httpx.Response(503 if failure == "503" else 200, text="PRIVATE invalid JSON")

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=1, max_input_bytes=64000)
    bound = budget.bind("p2-dev-001", "graphiti", "writer" if kind == "native" else "guard")
    transport_type = ReferenceTransport if kind == "native" else ProductTransport
    transport = transport_type(
        SecretStr("local-placeholder"), budget=bound, delegate=httpx.MockTransport(endpoint)
    )
    request = httpx.Request(
        "POST",
        "https://api.deepseek.com/chat/completions",
        json={"model": "deepseek-flash", "stream": False},
    )
    if kind == "product" and failure == "503":
        assert (await transport.handle_async_request(request)).status_code == 503
    else:
        error_type = (
            ReferenceCallError
            if kind == "native"
            else {
                "timeout": httpx.ReadTimeout,
                "network": httpx.ConnectError,
                "protocol": ValueError,
            }[failure]
        )
        with pytest.raises(error_type, match=code):
            if kind == "native":
                transport.handle_request(request)
            else:
                await transport.handle_async_request(request)
    record = budget.snapshot()["calls"][0]["record"]
    assert record["error_code"] == code and record["failure_phase"] == phase
    assert record["usage"] is None and "PRIVATE" not in json.dumps(record)
    if failure != "network":
        assert record["http_status"] == (503 if failure == "503" else 200)
    transport.close() if kind == "native" else await transport.shutdown()


def test_two_transports_share_durable_attempt_budget_and_preserve_usage(tmp_path):
    from hmph.eval_budget import RunBudget
    from hmph.reference_transport import ReferenceCallError, ReferenceTransport

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=1, max_input_bytes=10000)

    def endpoint(_):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    for index, stage in enumerate(("writer", "reader")):
        transport = ReferenceTransport(
            SecretStr("fake-private-key"),
            delegate=httpx.MockTransport(endpoint),
            budget=budget.bind("case-001", "mem0", stage),
        )
        with httpx.Client(transport=transport) as client:
            if index == 0:
                assert (
                    client.post(
                        "https://api.deepseek.com/chat/completions",
                        json={"model": "deepseek-flash"},
                    ).status_code
                    == 200
                )
            else:
                with pytest.raises(ReferenceCallError, match="reference_call_budget"):
                    client.post(
                        "https://api.deepseek.com/chat/completions",
                        json={"model": "deepseek-flash"},
                    )
    calls = budget.snapshot()["calls"]
    assert len(calls) == 1 and calls[0]["record"]["usage"]["total_tokens"] == 15


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


async def test_async_reference_transport_forwards_a_compressed_provider_response_once():
    import gzip

    from hmph.reference_transport import AsyncReferenceTransport

    payload = {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
    }
    transport = AsyncReferenceTransport(
        SecretStr("fake-private-key"),
        delegate=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-encoding": "gzip", "content-type": "application/json"},
                content=gzip.compress(json.dumps(payload).encode()),
            )
        ),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.post(
            "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
        )
        assert response.json() == payload
    assert transport.records[0]["usage"]["total_tokens"] == 105


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


async def test_async_reference_calls_share_the_budget_and_keep_native_credentials_unchanged():
    import asyncio

    from hmph.reference_transport import AsyncReferenceTransport, ReferenceCallError

    def endpoint(request):
        assert request.headers["Authorization"] == "Bearer test-private-key-not-real"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
            },
        )

    transport = AsyncReferenceTransport(
        SecretStr("test-private-key-not-real"), max_calls=2, delegate=httpx.MockTransport(endpoint)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        replies = await asyncio.gather(
            *(
                client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": "Bearer native-placeholder"},
                    json={"model": "deepseek-flash"},
                )
                for _ in range(3)
            ),
            return_exceptions=True,
        )
        completed = [r for r in replies if isinstance(r, httpx.Response)]
        assert len(completed) == 2
        assert sum(isinstance(r, ReferenceCallError) for r in replies) == 1
        assert all(r.json()["choices"][0]["message"]["content"] == "{}" for r in completed)
        assert all(
            r.request.headers["Authorization"] == "Bearer native-placeholder" for r in completed
        )
        with pytest.raises(ReferenceCallError, match="reference_destination_denied"):
            await client.post(
                "https://example.com/chat/completions", json={"model": "deepseek-flash"}
            )
    assert len(transport.records) == 2
    assert sum(r["usage"]["total_tokens"] for r in transport.records) == 210
    assert "test-private-key-not-real" not in json.dumps(transport.records)
