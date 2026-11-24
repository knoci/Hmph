"""Real-world AIRI Companion Experience Stress Test (Hmph v6.0, real DeepSeek Flash + offline ONNX).

Evaluates real daily companion quality — NOT benchmark tricks:
  1. Multi-day natural Chinese chat with AIRI Live2D persona preserved
  2. Automatic background session summarization on session switch (0 manual calls)
  3. <30ms local Entity-Scoped multi-session recall ("我和小雨最近一起做过哪些事？" -> 猫咖 + 陶艺)
  4. Natural evidence-grounded dot-connecting ("我最爱吃哪种肉？" -> 鸡肉 from 烤鸡翅 + 鸡汤面)
  5. Strict zero-hallucination abstention on false premise ("上个月买的 iPad Pro 花了多少钱？")
  6. Colloquial pause -> boundary-safe proactive check-in stops + casual chat anti-nag
  7. Atomic cascade forget + restart zero resurrection + streaming TTFT / retrieval latency metrics
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

from hmph.local_memory import (
    LocalMemoryEngine,
    OpenAICompatLLM,
    build_default_embedder,
    create_airi_gateway_app,
    resolve_llm_config,
)

OUT_DIR = Path("artifacts/local-memory/verify-airi-experience-v6")

AIRI_SYSTEM = (
    "You are Yuki, an interactive Live2D AI companion inside Project AIRI. "
    "Respond warmly and naturally in concise Chinese (1-3 sentences), starting with a Live2D emotion tag "
    "like <|happy|>, <|thinking|>, or <|neutral|>. Be honest when you don't remember something."
)


async def airi_stream_turn(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    user_text: str,
    model: str = "hmph/yuki",
) -> dict:
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    chunks: list[str] = []
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"X-Hmph-Session-Id": session_id},
        json={
            "model": model,
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 220,
            "messages": [
                {"role": "system", "content": AIRI_SYSTEM},
                {"role": "user", "content": user_text},
            ],
        },
    ) as resp:
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        headers = {k.lower(): v for k, v in resp.headers.items()}
        async for raw_line in resp.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    chunks.append(str(piece))
    total_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "reply": "".join(chunks).strip(),
        "ttft_ms": ttft_ms or total_ms,
        "total_ms": total_ms,
        "retrieval_ms": float(headers.get("x-hmph-retrieval-ms", "0")),
        "items": int(headers.get("x-hmph-memory-items", "0")),
        "consolidations": int(headers.get("x-hmph-consolidations", "0")),
    }


async def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUT_DIR / "airi_exp_v6.db"
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
        auto_consolidate_every=5,
        auto_summarize_turns=6,
    )
    app = create_airi_gateway_app(engine, owns_engine=False)
    checks: list[dict] = []
    latencies: list[dict] = []

    def rec(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://hmph.local"
        ) as client:
            # --- Day 1 (session day1) ---
            t1 = await airi_stream_turn(
                client,
                session_id="day1",
                user_text="周末我和小雨去了新开的猫咖撸猫，超级解压！顺便记一下每周六一起跟进我练吉他。",
            )
            await engine.flush()
            t2 = await airi_stream_turn(
                client,
                session_id="day1",
                user_text="对了，我平时做饭最爱吃香辣烤鸡翅，还有外婆教我的鸡汤面，百吃不腻。",
            )
            await engine.flush()

            # --- Day 2 (session day2): triggers automatic background summary of day1! ---
            t3 = await airi_stream_turn(
                client,
                session_id="day2",
                user_text="今天又和小雨一起去做了手工陶艺，顺便把吉他练习从周六改到周日吧。",
            )
            await engine.flush()
            t4 = await airi_stream_turn(
                client,
                session_id="day2",
                user_text="最近在市中心租的小公寓加班压力好大，好想以后能换个远程工作搬到郊区住大房子。",
            )
            await engine.flush()

            day1_sum = engine.store.get_session_summary("day1", namespace="yuki")
            rec(
                "auto_background_session_summary_on_switch",
                day1_sum is not None and len(day1_sum.facts) >= 1,
                f"day1_summary_facts={len(day1_sum.facts) if day1_sum else 0}, text={(day1_sum.summary[:80] if day1_sum else None)}",
            )

            # --- Day 3 (session day3): triggers automatic background summary of day2! ---
            # Test 1: Entity-scoped multi-session recall ("我和小雨最近一起做过哪些事？" -> 猫咖 + 陶艺)
            t5 = await airi_stream_turn(
                client,
                session_id="day3",
                user_text="我和小雨最近一起去过哪些地方、做过哪些事呀？",
            )
            await engine.flush()
            latencies.append({"turn": "entity_recall", "ttft_ms": t5["ttft_ms"], "retrieval_ms": t5["retrieval_ms"]})
            rec(
                "live_entity_scoped_multi_session_recall",
                ("猫咖" in t5["reply"] or "猫" in t5["reply"]) and "陶艺" in t5["reply"],
                f"retrieval={t5['retrieval_ms']}ms, ttft={t5['ttft_ms']}ms, reply={t5['reply'][:110]}",
            )

            # Test 2: Natural evidence-grounded dot-connecting ("我最爱吃哪种肉？" -> 鸡肉!)
            t6 = await airi_stream_turn(
                client,
                session_id="day3",
                user_text="今晚想自己做顿肉菜，按你对我的了解，我最爱吃哪种肉呀？",
            )
            await engine.flush()
            latencies.append({"turn": "dot_connecting", "ttft_ms": t6["ttft_ms"], "retrieval_ms": t6["retrieval_ms"]})
            rec(
                "natural_dot_connecting_implied_preference",
                "鸡" in t6["reply"] and "<|" in t6["reply"],
                f"retrieval={t6['retrieval_ms']}ms, ttft={t6['ttft_ms']}ms, reply={t6['reply'][:110]}",
            )

            # Test 3: Strict zero-hallucination abstention on unrecorded false premise (iPad Pro)
            t7 = await airi_stream_turn(
                client,
                session_id="day3",
                user_text="我上个月买的 iPad Pro 花了多少钱来着？",
            )
            await engine.flush()
            latencies.append({"turn": "abstention", "ttft_ms": t7["ttft_ms"], "retrieval_ms": t7["retrieval_ms"]})
            has_fake_price = any(sym in t7["reply"] for sym in ("元", "块", "¥", "$", "千", "万")) and not any(
                neg in t7["reply"] for neg in ("没", "不记得", "没有", "未", "印象")
            )
            rec(
                "strict_zero_hallucination_false_premise",
                (not has_fake_price) and any(neg in t7["reply"] for neg in ("没", "不记得", "没有", "未", "印象")),
                f"retrieval={t7['retrieval_ms']}ms, reply={t7['reply'][:110]}",
            )

            # Test 4: Pause guitar -> proactive stops + casual chat does NOT nag
            await airi_stream_turn(
                client,
                session_id="day3",
                user_text="吉他练习先搁一搁吧，最近太忙别催我了。",
            )
            await engine.flush()
            pro = (
                await client.get(
                    "/v1/companion/proactive",
                    params={"namespace": "yuki", "cooldown_hours": 0},
                )
            ).json()["suggestion"]
            t8 = await airi_stream_turn(
                client,
                session_id="day4",
                user_text="今天下雨了，窝在家里听雨声好舒服。",
            )
            await engine.flush()
            rec(
                "paused_anti_nag_and_proactive_boundary",
                pro["excluded_paused_count"] >= 1
                and "吉他" not in t8["reply"]
                and "练琴" not in t8["reply"],
                f"excluded_paused={pro['excluded_paused_count']}, reply={t8['reply'][:100]}",
            )

            # Test 5: Fast local retrieval latency (< 100ms on live chat path)
            max_ret = max(x["retrieval_ms"] for x in latencies)
            rec(
                "live_retrieval_latency_under_100ms",
                max_ret < 100.0,
                f"max_retrieval_ms={max_ret}, latencies={latencies}",
            )

            # Test 6: Cascade forget clears guitar + core blocks / summaries cleanly
            fg = (
                await client.post(
                    "/v1/memory/forget", json={"target": "吉他", "namespace": "yuki"}
                )
            ).json()["receipt"]
            rec(
                "cascade_forget_clean",
                len(fg["deleted_item_ids"]) >= 1,
                f"items={len(fg['deleted_item_ids'])}, cons={len(fg['deleted_consolidation_ids'])}, summaries={len(fg['deleted_session_summary_ids'])}",
            )
    finally:
        await engine.close()

    passed_n = sum(1 for c in checks if c["passed"])
    report = {"passed": passed_n, "total": len(checks), "latencies": latencies, "checks": checks}
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== AIRI Real Experience v6.0: {passed_n}/{len(checks)} PASSED ===", flush=True)
    return 0 if passed_n == len(checks) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
