"""Automated contract tests for Hmph v4.0 Headless AIRI Memory Gateway:
- Verifies zero-HTML headless positioning (GET / returns JSON descriptor)
- Verifies CORS preflight for AIRI Web / Tauri Desktop origins
- Verifies OpenAI-compatible POST /v1/chat/completions (streaming SSE & non-streaming)
  while preserving AIRI's original character/Live2D system prompt untouched
- Verifies strict multi-namespace character isolation (hmph/yuki vs hmph/ren)
- Verifies vector-indexed Sleep Consolidation, auto-consolidation trigger, and zero-resurrection forget
- Verifies MCP Streamable HTTP (POST /mcp) 7-tool roundtrip & REST API
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from hmph.local_memory import (
    ExtractedChange,
    ExtractionResponse,
    HashingEmbedder,
    LocalMemoryEngine,
    create_airi_gateway_app,
)
from hmph.local_memory.gateway import resolve_namespace_and_model


class AIRIGatewayMockLLM:
    model_name = "deepseek-flash"

    def __init__(self) -> None:
        self.captured_proxy_messages: list[list[dict]] = []
        self.captured_models: list[str] = []
        self.consolidation_calls = 0

    async def extract_changes(
        self,
        *,
        user_message,
        recent_messages,
        candidate_items,
        now_iso,
        timezone_name,
    ):
        piano = next((it for it in candidate_items if "练琴" in it.title), None)
        if "周六" in user_message and "练琴" in user_message:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="create",
                            title="每周练琴",
                            kind="plan",
                            summary="约好每周六练琴",
                            effective_time_text="每周六",
                            entities=["练琴", "周六", "古典吉他"],
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 35},
            )
        if "改到周日" in user_message and piano:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="reschedule",
                            target_item_id=piano.id,
                            title=piano.title,
                            summary="练琴改到周日",
                            effective_time_text="周日",
                            entities=["练琴", "周日"],
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 35},
            )
        if "跑步" in user_message:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="create",
                            title="晨跑计划",
                            kind="plan",
                            summary="每周三早晨跑步5公里",
                            effective_time_text="每周三早晨",
                            entities=["跑步"],
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 35},
            )
        return ExtractionResponse(changes=[]), {"total_tokens": 10}

    async def consolidate_insights(self, *, items, existing_consolidations, timezone_name):
        self.consolidation_calls += 1
        if not items:
            return [], {"total_tokens": 10}
        return (
            [
                {
                    "topic": "周末吉他练习习惯",
                    "summary": "用户偏好周末安排古典吉他练琴，最近已从周六调整至周日。",
                    "source_item_ids": [items[0].id],
                    "confidence": 0.93,
                }
            ],
            {"total_tokens": 40},
        )

    async def proxy_chat_completion(
        self, *, messages, model=None, temperature=None, max_tokens=None, extra_params=None
    ):
        self.captured_proxy_messages.append(messages)
        self.captured_models.append(model or self.model_name)
        sys_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        reply = f"[AIRI-NONSTREAM] has_memory={'<hmph_memory>' in sys_text}"
        return {
            "id": "chatcmpl-mock-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model or self.model_name,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

    async def proxy_chat_completion_stream(
        self, *, messages, model=None, temperature=None, max_tokens=None, extra_params=None
    ):
        self.captured_proxy_messages.append(messages)
        self.captured_models.append(model or self.model_name)
        for piece in ("<|happy|>哼，", "记下啦！"):
            yield {
                "id": "chatcmpl-mock-stream",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": model or self.model_name,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
        yield {
            "id": "chatcmpl-mock-stream",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": model or self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    async def generate_reply(self, **kwargs):
        return "ok", {"total_tokens": 10}


def test_resolve_namespace_and_model_patterns():
    assert resolve_namespace_and_model("hmph", None, "deepseek-flash") == ("default", "deepseek-flash")
    assert resolve_namespace_and_model("hmph/yuki", None, "deepseek-flash") == ("yuki", "deepseek-flash")
    assert resolve_namespace_and_model("hmph/yuki@deepseek-chat", None, "deepseek-flash") == (
        "yuki",
        "deepseek-chat",
    )
    assert resolve_namespace_and_model("deepseek-chat", "airi-ren", "deepseek-flash") == (
        "airi-ren",
        "deepseek-chat",
    )


def test_airi_headless_gateway_openai_cors_namespaces_mcp_and_forget(tmp_path):
    db_path = tmp_path / "airi_v4.db"
    llm = AIRIGatewayMockLLM()
    # Set auto_consolidate_every=2 so 2 mutations automatically trigger Sleep Consolidation!
    engine = LocalMemoryEngine(
        db_path,
        llm=llm,
        embedder=HashingEmbedder(),
        auto_consolidate_every=2,
    )
    app = create_airi_gateway_app(engine, owns_engine=True)

    airi_system = "You are Yuki, an AIRI Live2D companion. Always emit emotion tags like <|happy|>."

    with TestClient(app) as client:
        # 1. Headless verification: GET / returns JSON descriptor, zero HTML Web UI.
        root = client.get("/")
        assert root.status_code == 200
        assert "application/json" in root.headers["content-type"]
        assert root.json()["mode"] == "headless-memory-component"

        # 2. CORS preflight for AIRI Web origin.
        opt = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "https://airi.moeru.ai",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,authorization,x-hmph-namespace",
            },
        )
        assert opt.status_code == 200
        assert opt.headers.get("access-control-allow-origin") == "*"

        # 3. OpenAI Streaming turn in namespace "yuki" (via model="hmph/yuki").
        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "hmph/yuki",
                "stream": True,
                "messages": [
                    {"role": "system", "content": airi_system},
                    {"role": "user", "content": "记一下，每周六练琴"},
                ],
            },
        )
        assert r1.status_code == 200
        assert r1.headers["x-hmph-namespace"] == "yuki"
        assert "chat.completion.chunk" in r1.text
        assert "data: [DONE]" in r1.text
        # Original AIRI Live2D system prompt MUST be preserved!
        assert airi_system in llm.captured_proxy_messages[0][0]["content"]
        assert llm.captured_models[0] == "deepseek-flash"

        # 4. Second streaming turn reschedules -> triggers auto sleep consolidation (2nd mutation).
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "hmph/yuki",
                "stream": True,
                "messages": [
                    {"role": "system", "content": airi_system},
                    {"role": "user", "content": "练琴改到周日吧"},
                ],
            },
        )
        assert r2.status_code == 200

        # Trigger non-streaming query in "yuki" to inspect injected <hmph_memory> & auto-consolidation.
        # Flush background tasks via /v1/memory/consolidate or read after short sync.
        mem_yuki = client.get("/v1/memory", params={"namespace": "yuki"}).json()
        assert len(mem_yuki["active"]) == 1
        assert len(mem_yuki["active"][0]["versions"]) == 2
        assert len(mem_yuki["consolidations"]) == 1
        assert llm.consolidation_calls >= 1

        r3 = client.post(
            "/v1/chat/completions",
            json={
                "model": "hmph/yuki",
                "stream": False,
                "messages": [
                    {"role": "system", "content": airi_system},
                    {"role": "user", "content": "我现在练琴约在周几？有什么建议？"},
                ],
            },
        )
        assert r3.status_code == 200
        assert r3.json()["object"] == "chat.completion"
        injected_sys = llm.captured_proxy_messages[-1][0]["content"]
        assert airi_system in injected_sys
        assert "<hmph_memory>" in injected_sys
        assert "周日" in injected_sys
        assert "周末吉他练习习惯" in injected_sys

        # 5. Strict multi-namespace isolation: character "ren" MUST NOT see "yuki"'s piano memory!
        r_ren = client.post(
            "/v1/chat/completions",
            json={
                "model": "hmph/ren",
                "stream": False,
                "messages": [
                    {"role": "system", "content": "You are Ren."},
                    {"role": "user", "content": "每周三早晨跑步"},
                ],
            },
        )
        assert r_ren.status_code == 200
        assert r_ren.headers["x-hmph-namespace"] == "ren"
        assert "练琴" not in llm.captured_proxy_messages[-1][0]["content"]

        models = [m["id"] for m in client.get("/v1/models").json()["data"]]
        assert "hmph" in models and "hmph/yuki" in models and "hmph/ren" in models

        # 6. MCP Streamable HTTP (POST /mcp) 8-tool verification & namespace isolation.
        mcp_tools = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        ).json()
        tool_names = {t["name"] for t in mcp_tools["result"]["tools"]}
        assert tool_names == {
            "hmph_remember",
            "hmph_recall",
            "hmph_list_memories",
            "hmph_forget",
            "hmph_consolidate",
            "hmph_graph",
            "hmph_toggle_pause",
            "hmph_core_memory",
        }

        # 6b. Core memory block REST + proactive check-in + retry idempotency + SSE event check.
        put_core = client.put(
            "/v1/memory/core/relationship_style",
            json={"namespace": "yuki", "content": "喜欢轻松温柔带一点吐槽的交流方式"},
        )
        assert put_core.status_code == 200
        assert len(client.get("/v1/memory/core", params={"namespace": "yuki"}).json()["blocks"]) >= 1

        pro = client.get(
            "/v1/companion/proactive",
            params={"namespace": "yuki", "cooldown_hours": 0},
        ).json()["suggestion"]
        assert pro["should_initiate"] is True
        assert pro["reason"] == "active_plan_checkin"

        # Retry idempotency: sending identical turn twice inside 20s does NOT duplicate user message.
        before_msgs = len(engine.get_session_messages("idemp-s1", namespace="ren"))
        for _ in range(2):
            client.post(
                "/v1/chat/completions",
                headers={"X-Hmph-Session-Id": "idemp-s1"},
                json={
                    "model": "hmph/ren",
                    "stream": False,
                    "messages": [{"role": "user", "content": "测试重试幂等消息"}],
                },
            )
        after_user_msgs = [
            m
            for m in engine.get_session_messages("idemp-s1", namespace="ren")
            if m.role == "user"
        ]
        assert len(after_user_msgs) == before_msgs + 1

        # SSE /v1/memory/events emits connected frame cleanly.
        sse_resp = client.get("/v1/memory/events", params={"namespace": "yuki", "max_events": 0})
        assert sse_resp.status_code == 200
        assert "event: connected" in sse_resp.text

        mcp_graph = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "hmph_graph", "arguments": {"namespace": "yuki"}},
            },
        ).json()
        graph_payload = json.loads(mcp_graph["result"]["content"][0]["text"])
        assert graph_payload["namespace"] == "yuki"
        assert graph_payload["stats"]["items"] == 1
        assert graph_payload["stats"]["consolidations"] == 1

        # 7. Cascade forget in "yuki" deletes piano + consolidation, leaving "ren" untouched!
        fg = client.post(
            "/v1/memory/forget",
            json={"target": "练琴", "namespace": "yuki"},
        ).json()["receipt"]
        assert len(fg["deleted_item_ids"]) == 1
        assert len(fg["deleted_consolidation_ids"]) == 1

        assert client.get("/v1/memory", params={"namespace": "yuki"}).json()["active"] == []
        assert client.get("/v1/memory", params={"namespace": "yuki"}).json()["consolidations"] == []
        assert len(client.get("/v1/memory", params={"namespace": "ren"}).json()["active"]) == 1

    # 8. Restart persistence check on same SQLite file: "yuki" stays empty (zero resurrection), "ren" survives.
    engine2 = LocalMemoryEngine(db_path, llm=llm, embedder=HashingEmbedder())
    app2 = create_airi_gateway_app(engine2, owns_engine=True)
    with TestClient(app2) as client2:
        assert client2.get("/v1/memory", params={"namespace": "yuki"}).json()["active"] == []
        assert client2.get("/v1/memory", params={"namespace": "yuki"}).json()["consolidations"] == []
        assert len(client2.get("/v1/memory", params={"namespace": "ren"}).json()["active"]) == 1
