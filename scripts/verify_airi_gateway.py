"""Real end-to-end verification of Hmph v4.0 Headless AIRI Memory Gateway with real DeepSeek Flash & ONNX:
Simulates Project AIRI calling POST /v1/chat/completions (streaming SSE) + POST /mcp + REST API:
  1. Headless descriptor check (GET / -> JSON, zero HTML) + CORS preflight
  2. AIRI Character 'yuki' (model='hmph/yuki'): remember Saturday piano + classical guitar preference
  3. Colloquial reschedule Saturday -> Sunday (preserves AIRI Live2D emotion tags)
  4. Colloquial pause ('先搁一搁，这阵子别催了') -> unrelated weather chat (anti-nag check)
  5. Switch AIRI Character 'ren' (model='hmph/ren'): ask about piano -> proves zero cross-character leakage
  6. Sleep Consolidation via MCP HTTP (POST /mcp hmph_consolidate) -> vector-indexed insight recalled in 'yuki'
  7. Atomic Cascade Forget in 'yuki' -> restart brand-new engine on same SQLite DB -> zero resurrection check
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

from hmph.local_memory import (
    LocalMemoryEngine,
    OpenAICompatLLM,
    build_default_embedder,
    create_airi_gateway_app,
    resolve_llm_config,
)

OUT_DIR = Path("artifacts/local-memory/verify-airi-v4")

AIRI_YUKI_PROMPT = (
    "You are Yuki, an interactive Live2D companion running inside Project AIRI. "
    "Speak naturally in concise Chinese (<=2 sentences) and start every reply with an emotion tag "
    "such as <|happy|>, <|neutral|>, or <|shy|>."
)
AIRI_REN_PROMPT = (
    "You are Ren, a calm minimalist Live2D companion inside Project AIRI. "
    "Speak concisely in Chinese and NEVER invent memories you do not have."
)


async def stream_openai_chat(
    client: httpx.AsyncClient,
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    session_id: str,
) -> tuple[str, dict[str, str]]:
    chunks: list[str] = []
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Hmph-Session-Id": session_id},
        json={
            "model": model,
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 180,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        },
    ) as resp:
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        headers = {k.lower(): v for k, v in resp.headers.items()}
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    chunks.append(str(piece))
    return "".join(chunks).strip(), headers


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUT_DIR / "airi_e2e.db"
    for ext in ("", "-wal", "-shm"):
        p = Path(f"{db_path}{ext}")
        if p.exists():
            p.unlink()

    api_key, base_url, model, tz = resolve_llm_config()
    llm = OpenAICompatLLM(api_key, base_url=base_url, model=model, temperature=0.2)
    embedder = build_default_embedder()
    engine = LocalMemoryEngine(
        db_path,
        llm=llm,
        embedder=embedder,
        timezone_name=tz,
        auto_consolidate_every=0,
    )
    app = create_airi_gateway_app(engine, owns_engine=False)
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}", flush=True)

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://hmph.local") as client:
            # 1. Headless JSON root & CORS preflight
            r_root = await client.get("/")
            record(
                "headless_json_root",
                r_root.status_code == 200
                and "application/json" in r_root.headers.get("content-type", "")
                and r_root.json().get("mode") == "headless-memory-component",
                f"mode={r_root.json().get('mode')}",
            )
            r_opt = await client.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "https://airi.moeru.ai",
                    "Access-Control-Request-Method": "POST",
                },
            )
            record(
                "cors_airi_web_preflight",
                r_opt.status_code == 200 and r_opt.headers.get("access-control-allow-origin") == "*",
                f"acao={r_opt.headers.get('access-control-allow-origin')}",
            )

            # 2. AIRI Yuki Turn 1 (streaming)
            reply1, h1 = await stream_openai_chat(
                client,
                model="hmph/yuki",
                system_prompt=AIRI_YUKI_PROMPT,
                user_text="记一下，我们约好每周六一起跟进练琴进展，我平时特别喜欢古典和指弹吉他。",
                session_id="airi-yuki-s1",
            )
            await engine.flush()
            record(
                "yuki_turn1_stream_and_persona_preserved",
                h1.get("x-hmph-namespace") == "yuki" and "<|" in reply1 and len(engine.list_items(namespace="yuki")) >= 1,
                f"ns={h1.get('x-hmph-namespace')}, items={len(engine.list_items(namespace='yuki'))}, reply={reply1[:80]}",
            )

            # 3. AIRI Yuki Turn 2: colloquial reschedule Saturday -> Sunday
            reply2, _ = await stream_openai_chat(
                client,
                model="hmph/yuki",
                system_prompt=AIRI_YUKI_PROMPT,
                user_text="这周起练琴改到周日吧，周六我经常起不来。",
                session_id="airi-yuki-s1",
            )
            await engine.flush()
            yuki_items = engine.list_items(namespace="yuki")
            piano_item = next((it for it in yuki_items if "练琴" in it.title or "琴" in it.title), yuki_items[0])
            record(
                "yuki_turn2_reschedule_evolution",
                len(piano_item.versions) >= 2 and ("周日" in (piano_item.current.summary if piano_item.current else "") or "周日" in (piano_item.current.effective_time_text or "")),
                f"versions={len(piano_item.versions)}, latest={piano_item.current.summary if piano_item.current else ''}",
            )

            # 4. AIRI Yuki Turn 3: Sleep Consolidation via MCP HTTP (POST /mcp) + Core Memory auto-sync
            r_mcp = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {"name": "hmph_consolidate", "arguments": {"namespace": "yuki"}},
                },
            )
            cons_yuki = engine.list_consolidations(namespace="yuki")
            core_yuki = engine.get_core_memory_blocks(namespace="yuki")
            record(
                "mcp_http_consolidate_and_core_memory_sync",
                r_mcp.status_code == 200 and len(cons_yuki) >= 1 and len(core_yuki) >= 1,
                f"consolidations={len(cons_yuki)}, core_blocks={len(core_yuki)}, topic={cons_yuki[0].topic if cons_yuki else None}",
            )

            # 4b. Proactive check-in triggers for active plan BEFORE pause
            r_pro1 = await client.get(
                "/v1/companion/proactive",
                params={"namespace": "yuki", "cooldown_hours": 0},
            )
            pro1 = r_pro1.json()["suggestion"]
            record(
                "proactive_active_plan_checkin",
                pro1["should_initiate"] is True and pro1["reason"] == "active_plan_checkin",
                f"reason={pro1['reason']}, topic={pro1.get('topic')}",
            )

            # 5. AIRI Yuki Turn 4: colloquial pause -> proactive MUST stop + unrelated chat anti-nag
            await stream_openai_chat(
                client,
                model="hmph/yuki",
                system_prompt=AIRI_YUKI_PROMPT,
                user_text="练琴先搁一搁吧，这阵子别继续催我了。",
                session_id="airi-yuki-s1",
            )
            await engine.flush()
            r_pro2 = await client.get(
                "/v1/companion/proactive",
                params={"namespace": "yuki", "cooldown_hours": 0},
            )
            pro2 = r_pro2.json()["suggestion"]
            record(
                "proactive_respects_paused_boundary",
                pro2["should_initiate"] is False and pro2["excluded_paused_count"] >= 1,
                f"should_initiate={pro2['should_initiate']}, excluded_paused={pro2['excluded_paused_count']}",
            )
            reply_weather, h_w = await stream_openai_chat(
                client,
                model="hmph/yuki",
                system_prompt=AIRI_YUKI_PROMPT,
                user_text="今天天气真不错，中午想吃拉面。",
                session_id="airi-yuki-s2",
            )
            await engine.flush()
            record(
                "yuki_paused_anti_nag_in_new_session",
                "练琴" not in reply_weather and int(h_w.get("x-hmph-memory-items", "0")) == 0,
                f"injected_items={h_w.get('x-hmph-memory-items')}, reply={reply_weather[:80]}",
            )

            # 6. Multi-character namespace isolation: switch AIRI to 'hmph/ren'
            reply_ren, h_ren = await stream_openai_chat(
                client,
                model="hmph/ren",
                system_prompt=AIRI_REN_PROMPT,
                user_text="我们约好周几练琴来着？",
                session_id="airi-ren-s1",
            )
            await engine.flush()
            record(
                "multi_character_namespace_isolation",
                h_ren.get("x-hmph-namespace") == "ren"
                and "周日" not in reply_ren
                and "周六" not in reply_ren
                and len(engine.list_items(namespace="ren")) == 0,
                f"ren_items={len(engine.list_items(namespace='ren'))}, reply={reply_ren[:80]}",
            )

            # 7. Atomic Cascade Forget in 'yuki' via REST API -> closes & restarts fresh engine
            r_fg = await client.post(
                "/v1/memory/forget",
                json={"target": "练琴", "namespace": "yuki"},
            )
            fg = r_fg.json()["receipt"]
            record(
                "cascade_forget_clears_items_and_consolidations",
                len(fg["deleted_item_ids"]) >= 1 and len(fg["deleted_consolidation_ids"]) >= 1,
                f"items={len(fg['deleted_item_ids'])}, cons={len(fg['deleted_consolidation_ids'])}, msgs={len(fg['deleted_message_ids'])}",
            )
    finally:
        await engine.close()

    # 8. Brand-new engine on same SQLite file -> verify zero resurrection!
    llm2 = OpenAICompatLLM(api_key, base_url=base_url, model=model, temperature=0.2)
    eng2 = LocalMemoryEngine(db_path, llm=llm2, embedder=embedder, timezone_name=tz)
    try:
        app2 = create_airi_gateway_app(eng2, owns_engine=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app2), base_url="http://hmph.local"
        ) as client2:
            reply_after, _ = await stream_openai_chat(
                client2,
                model="hmph/yuki",
                system_prompt=AIRI_YUKI_PROMPT,
                user_text="练琴原来约的是哪天？现在改到哪天了？",
                session_id="airi-yuki-after-restart",
            )
            record(
                "restart_zero_resurrection",
                len(eng2.list_consolidations(namespace="yuki")) == 0
                and "周六" not in reply_after
                and "周日" not in reply_after,
                f"cons_after={len(eng2.list_consolidations(namespace='yuki'))}, reply={reply_after[:90]}",
            )
    finally:
        await eng2.close()

    passed_n = sum(1 for c in checks if c["passed"])
    report = {"passed": passed_n, "total": len(checks), "checks": checks}
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== AIRI Gateway v4.0 E2E: {passed_n}/{len(checks)} PASSED ===", flush=True)
    return 0 if passed_n == len(checks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
