"""Interactive command-line chat for the standalone single-file SQLite memory backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from hmph.local_memory.engine import LocalMemoryEngine
from hmph.local_memory.types import ItemRecord, TurnResult


def format_items_report(items: list[ItemRecord]) -> str:
    if not items:
        return "（目前还没有记住任何长期事项。）"
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        cur = item.current
        time_info = f" | 时间: {cur.effective_time_text}" if (cur and cur.effective_time_text) else ""
        lines.append(
            f"{idx}. [{item.status}] {item.title} ({item.kind}, id={item.id[:8]}){time_info}"
        )
        for ver in item.versions:
            t = f" [{ver.effective_time_text}]" if ver.effective_time_text else ""
            c = f" [条件: {ver.condition_text}]" if ver.condition_text else ""
            lines.append(
                f"   - v{ver.version_no} {ver.change_type} -> {ver.status}: {ver.summary}{t}{c}"
            )
            lines.append(
                f"     原话({ver.source_message_id[:8]}): 「{ver.source_quote}」"
            )
    return "\n".join(lines)


def format_turn_inspect(turn: TurnResult) -> str:
    ev = turn.evidence
    lines = [
        f"意图: {ev.query_intent} | 向量引擎: {ev.embedding_backend} | 检索: {ev.retrieval_ms}ms | 生成: {turn.generation_ms}ms",
        f"引用事项数: {len(ev.used_item_ids)} | 引用原话数: {len(ev.used_source_message_ids)} | 待处理原话: {len(ev.pending_message_ids)}",
    ]
    if ev.narrated_context:
        lines.append("--- 交给角色的前情 ---")
        lines.append(ev.narrated_context)
    return "\n".join(lines)


async def run_local_chat(
    db_path: Path | str = Path("artifacts/local-memory/hmph.db"),
    *,
    session_id: str | None = None,
) -> None:
    engine = LocalMemoryEngine.from_env(db_path)
    active_session = session_id or f"session-{uuid4().hex[:8]}"
    last_turn: TurnResult | None = None

    print(f"Hmph 本地 SQLite 记忆后端已启动 | DB: {Path(db_path).resolve()}")
    print(f"模型: {engine.llm.model_name} | 向量: {engine.embedder.name} | 会话: {active_session}")
    print("命令: /new 新会话 | /memory 查看记忆与原话 | /inspect 查看上轮前情 | /forget <关键词或ID> 遗忘 | /quit 退出\n")

    try:
        while True:
            try:
                raw = await asyncio.to_thread(input, "你 > ")
            except EOFError:
                break
            text = raw.strip()
            if not text:
                continue
            if text in {"/quit", "/exit", "exit", "quit"}:
                break
            if text == "/new":
                await engine.flush()
                active_session = f"session-{uuid4().hex[:8]}"
                print(f"[已切换到新会话: {active_session}]\n")
                continue
            if text == "/memory":
                await engine.flush()
                print(format_items_report(engine.list_items()))
                ambs = engine.list_open_ambiguities()
                if ambs:
                    print("\n未决指代疑点:")
                    for amb in ambs:
                        print(f"  - 原话「{amb.source_quote}」| {amb.note}")
                print()
                continue
            if text == "/inspect":
                if last_turn is None:
                    print("[还没有完成过对话回合]\n")
                else:
                    print(format_turn_inspect(last_turn) + "\n")
                continue
            if text.startswith("/forget"):
                target = text[len("/forget") :].strip()
                if not target:
                    print("用法: /forget <事项关键词或ID>\n")
                    continue
                receipt = await engine.forget(target)
                print(
                    f"[已遗忘「{receipt.query}」: 清理事项 {len(receipt.deleted_item_ids)} 项, "
                    f"版本 {len(receipt.deleted_version_ids)} 条, 相关消息 {len(receipt.deleted_message_ids)} 条, "
                    f"图边 {receipt.deleted_edge_count} 条]\n"
                )
                continue

            last_turn = await engine.chat(active_session, text)
            print(f"Hmph > {last_turn.reply}\n")
    except KeyboardInterrupt:
        print("\n[收到中断信号，正在等待后台记忆落盘...]")
    finally:
        await engine.close()
        print("[后台记忆已同步落盘，连接已关闭。]")


def run_airi_gateway(
    db_path: Path | str = Path("artifacts/local-memory/hmph.db"),
    *,
    host: str = "127.0.0.1",
    port: int = 8018,
    auto_consolidate_every: int = 12,
) -> None:
    """Run the headless Dual-Protocol AIRI Memory Gateway (OpenAI-Compatible + MCP HTTP + REST)."""
    import uvicorn

    from hmph.local_memory.gateway import create_airi_gateway_app

    engine = LocalMemoryEngine.from_env(db_path)
    engine.auto_consolidate_every = max(0, int(auto_consolidate_every))
    app = create_airi_gateway_app(engine, owns_engine=True)
    print(f"Hmph v4.0 Headless Memory Gateway 已启动 | http://{host}:{port}")
    print(f"  • AIRI OpenAI-Compatible Base URL : http://{host}:{port}/v1")
    print(f"  • MCP HTTP JSON-RPC Endpoint      : http://{host}:{port}/mcp")
    print(f"  • Headless Memory REST API        : http://{host}:{port}/v1/memory")
    print(f"  • SQLite DB                       : {Path(db_path).resolve()}")
    print(f"  • Upstream LLM / Embedder         : {engine.llm.model_name} / {engine.embedder.name}")
    print("按 Ctrl+C 优雅落盘并停止服务。")
    uvicorn.run(app, host=host, port=port, access_log=False, log_level="warning")
