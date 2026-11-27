"""Interactive command-line chat for the standalone SQLite memory backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from hmph.local_memory.engine import LocalMemoryEngine
from hmph.local_memory.types import ItemRecord, TurnResult


def format_items_report(items: list[ItemRecord], *, locale: str = "en") -> str:
    zh = locale == "zh"
    if not items:
        return "（目前还没有记住任何长期事项。）" if zh else "(No long-term items yet.)"
    time_label = "时间" if zh else "Time"
    condition_label = "条件" if zh else "Condition"
    source_label = "原话" if zh else "Source"
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        cur = item.current
        time_info = f" | {time_label}: {cur.effective_time_text}" if (cur and cur.effective_time_text) else ""
        lines.append(
            f"{idx}. [{item.status}] {item.title} ({item.kind}, id={item.id[:8]}){time_info}"
        )
        for ver in item.versions:
            time_text = f" [{ver.effective_time_text}]" if ver.effective_time_text else ""
            condition = f" [{condition_label}: {ver.condition_text}]" if ver.condition_text else ""
            lines.append(
                f"   - {ver.change_type} -> {ver.status}: {ver.summary}{time_text}{condition}"
            )
            lines.append(f"     {source_label}({ver.source_message_id[:8]}): {ver.source_quote}")
    return "\n".join(lines)


def format_turn_inspect(turn: TurnResult, *, locale: str = "en") -> str:
    ev = turn.evidence
    if locale == "zh":
        lines = [
            f"意图: {ev.query_intent} | 向量引擎: {ev.embedding_backend} | 检索: {ev.retrieval_ms}ms | 生成: {turn.generation_ms}ms",
            f"引用事项数: {len(ev.used_item_ids)} | 引用原话数: {len(ev.used_source_message_ids)} | 待处理原话: {len(ev.pending_message_ids)}",
        ]
    else:
        lines = [
            f"Intent: {ev.query_intent} | Embedder: {ev.embedding_backend} | Retrieval: {ev.retrieval_ms}ms | Generation: {turn.generation_ms}ms",
            f"Items used: {len(ev.used_item_ids)} | Sources used: {len(ev.used_source_message_ids)} | Pending messages: {len(ev.pending_message_ids)}",
        ]
    if ev.narrated_context:
        lines.append("--- 交给角色的前情 ---" if locale == "zh" else "--- Context sent to the model ---")
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

    def localize(en: str, zh: str) -> str:
        return zh if engine.locale == "zh" else en

    print(f"{localize('Hmph SQLite memory is ready', 'Hmph 本地 SQLite 记忆后端已启动')} | DB: {db_path}")
    print(f"{localize('Model', '模型')}: {engine.llm.model_name} | "
          f"{localize('Embedder', '向量')}: {engine.embedder.name} | "
          f"{localize('Session', '会话')}: {active_session}")
    print(localize(
        "Commands: /new new session | /memory inspect items and sources | /inspect last context | /forget <term or ID> | /quit\n",
        "命令: /new 新会话 | /memory 查看记忆与原话 | /inspect 查看上轮前情 | /forget <关键词或ID> 遗忘 | /quit 退出\n",
    ))

    try:
        await engine.recover_pending_extractions()
        while True:
            try:
                raw = await asyncio.to_thread(input, localize("You > ", "你 > "))
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
                print(localize(f"[New session: {active_session}]\n", f"[已切换到新会话: {active_session}]\n"))
                continue
            if text == "/memory":
                await engine.flush()
                print(format_items_report(engine.list_items(), locale=engine.locale))
                ambs = engine.list_open_ambiguities()
                if ambs:
                    print(localize("\nUnresolved references:", "\n未决指代疑点:"))
                    for amb in ambs:
                        print(f"  - {localize('Source', '原话')}: {amb.source_quote} | {amb.note}")
                print()
                continue
            if text == "/inspect":
                if last_turn is None:
                    print(localize("[No completed turn yet]\n", "[还没有完成过对话回合]\n"))
                else:
                    print(format_turn_inspect(last_turn, locale=engine.locale) + "\n")
                continue
            if text.startswith("/forget"):
                target = text[len("/forget") :].strip()
                if not target:
                    print(localize("Usage: /forget <term or ID>\n", "用法: /forget <事项关键词或ID>\n"))
                    continue
                receipt = await engine.forget(target)
                items = len(receipt.deleted_item_ids)
                versions = len(receipt.deleted_version_ids)
                messages = len(receipt.deleted_message_ids)
                print(localize(
                    f"[Forgot {receipt.query}: {items} items, {versions} versions, "
                    f"{messages} messages, {receipt.deleted_edge_count} graph edges]\n",
                    f"[已遗忘「{receipt.query}」: 清理事项 {items} 项, 版本 {versions} 条, "
                    f"相关消息 {messages} 条, 图边 {receipt.deleted_edge_count} 条]\n",
                ))
                continue

            last_turn = await engine.chat(active_session, text)
            print(f"Hmph > {last_turn.reply}\n")
    except KeyboardInterrupt:
        print(localize("\n[Waiting for memory writes before exit...]", "\n[收到中断信号，正在等待后台记忆落盘...]"))
    finally:
        await engine.close()
        print(localize("[Memory connection closed.]", "[后台记忆已同步落盘，连接已关闭。]"))


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
    zh = engine.locale == "zh"
    heading = "Hmph 无头记忆网关已启动" if zh else "Hmph Headless Memory Gateway is ready"
    print(f"{heading} | http://{host}:{port}")
    print(f"  • AIRI OpenAI-Compatible Base URL : http://{host}:{port}/v1")
    print(f"  • MCP HTTP JSON-RPC Endpoint      : http://{host}:{port}/mcp")
    print(f"  • Headless Memory REST / SSE API  : http://{host}:{port}/v1/memory")
    print(f"  • SQLite DB                       : {Path(db_path).resolve()}")
    print(f"  • Upstream LLM / Embedder         : {engine.llm.model_name} / {engine.embedder.name}")
    print("按 Ctrl+C 落盘并停止服务。" if zh else "Press Ctrl+C to save and shut down.")
    uvicorn.run(app, host=host, port=port, access_log=False, log_level="warning")
