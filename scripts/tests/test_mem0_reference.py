"""Optional native integration: run with artifacts/mem0-env and PYTHONPATH=src."""

import json
from importlib.util import find_spec
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from openai import OpenAI
from pydantic import SecretStr

from hmph.reference_transport import ReferenceTransport

if find_spec("mem0") is None:
    pytest.skip("Run native Mem0 tests in artifacts/mem0-env", allow_module_level=True)


def test_native_mem0_pgvector_crud_isolation_and_history_cleanup():
    from hmph.mem0_reference import native_mem0

    def endpoint(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"memory": [{"text": "User works at Cedar Atelier."}]}
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    transport = ReferenceTransport(
        SecretStr("fake-local-test-key"), delegate=httpx.MockTransport(endpoint)
    )
    directory = Path("artifacts/references/runs") / ("mem0-test-" + uuid4().hex)
    with httpx.Client(transport=transport) as http:
        sdk = OpenAI(
            api_key="native-placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        with native_mem0(directory, sdk, model="deepseek-flash") as native:
            result = native.add("I work at Cedar Atelier.", user_id="alpha")
            memory_id = result["results"][0]["id"]
            assert native.search("employer", filters={"user_id": "alpha"})["results"]
            assert not native.search("employer", filters={"user_id": "beta"})["results"]
            native.update(memory_id, text="User now works at Juniper Works.")
            hits = native.search("employer", filters={"user_id": "alpha"})["results"]
            assert any("Juniper Works" in hit["memory"] for hit in hits)
            assert not any("Cedar Atelier" in hit["memory"] for hit in hits)
            native.delete(memory_id)
            assert not native.search("employer", filters={"user_id": "alpha"})["results"]
            # Native delete retains old text in history: distinguish it from complete forgetting.
            assert "Juniper Works" in json.dumps(native.history(memory_id))
    assert not (directory / "history.db").exists()
    assert len(transport.records) == 1 and transport.records[0]["usage"]["total_tokens"] == 120


def test_provider_failure_still_cleans_the_native_schema_and_history():
    from hmph.mem0_reference import native_mem0

    transport = ReferenceTransport(
        SecretStr("fake-local-test-key"),
        delegate=httpx.MockTransport(
            lambda _: httpx.Response(500, json={"error": "fake-local-test-key"})
        ),
    )
    directory = Path("artifacts/references/runs") / ("mem0-test-" + uuid4().hex)
    audit = {}
    with httpx.Client(transport=transport) as http:
        sdk = OpenAI(
            api_key="native-placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        with pytest.raises(Exception, match="LLM extraction failed") as error:
            with native_mem0(directory, sdk, model="deepseek-flash", cleanup_audit=audit) as native:
                native.add("I work at Cedar Atelier.", user_id="alpha")
    assert audit == {"schema_removed": True, "sqlite_history_removed": True}
    assert not (directory / "history.db").exists()
    assert len(transport.records) == 1 and transport.records[0]["usage"] is None
    assert "fake-local-test-key" not in json.dumps(transport.records)
    assert "fake-local-test-key" not in str(error.value)


def test_rebuild_from_retained_sources_cannot_read_the_retired_history():
    from hmph.mem0_reference import native_mem0

    prompts = []

    def endpoint(request):
        prompts.append(json.loads(request.content)["messages"][-1]["content"])
        memory = "User's name is Mira." if len(prompts) == 1 else "User works at Juniper Works."
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": json.dumps({"memory": [{"text": memory}]})},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    transport = ReferenceTransport(
        SecretStr("fake-local-test-key"), delegate=httpx.MockTransport(endpoint)
    )
    root = Path("artifacts/references/runs") / ("mem0-test-" + uuid4().hex)
    with httpx.Client(transport=transport) as http:
        sdk = OpenAI(
            api_key="native-placeholder",
            base_url="https://api.deepseek.com",
            http_client=http,
            max_retries=0,
        )
        with native_mem0(root / "old", sdk, model="deepseek-flash") as old:
            old.add("My name is Mira.", user_id="alpha")
        # Replay only the retained source into a new namespace after complete retirement.
        with native_mem0(root / "new", sdk, model="deepseek-flash") as new:
            new.add("I work at Juniper Works.", user_id="alpha")
            hits = new.get_all(filters={"user_id": "alpha"})["results"]
            assert len(hits) == 1 and "Juniper Works" in hits[0]["memory"]
        assert "Mira" not in prompts[1]
    assert not (root / "old/history.db").exists()
    assert not (root / "new/history.db").exists()
