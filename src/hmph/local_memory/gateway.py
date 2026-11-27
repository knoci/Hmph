"""Headless Dual-Protocol AIRI Memory Gateway (OpenAI-Compatible Proxy + MCP HTTP + Memory REST API).

Designed specifically as a drop-in memory component for Project AIRI (Web & Desktop/Tauri)
and any OpenAI-compatible or MCP AI companion/agent. No HTML/Web UI is served.

Protocols exposed on one single-process SQLite daemon:
  1. OpenAI-Compatible Memory Proxy (with CORS):
       - GET  /v1/models
       - POST /v1/chat/completions  (streaming SSE & non-streaming JSON, preserves AIRI system prompt)
  2. MCP Streamable HTTP JSON-RPC 2.0:
       - POST /mcp  (7 namespace-aware tools shared with `hmph local-mcp` stdio)
  3. Headless Memory REST API:
       - GET  /                     (JSON service descriptor)
       - GET  /v1/health
       - GET  /v1/memory
       - GET  /v1/memory/namespaces
       - GET  /v1/memory/graph
       - POST /v1/memory/recall
       - POST /v1/memory/remember
       - POST /v1/memory/consolidate
       - POST /v1/memory/forget
       - POST /v1/memory/items/{item_id}/toggle
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from hmph import __version__
from hmph.local_memory.engine import LocalMemoryEngine
from hmph.local_memory.mcp_server import LocalMCPServer
from hmph.local_memory.store import normalize_namespace


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(default="hmph")
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None


class RecallRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    namespace: str = Field(default="default", max_length=64)
    session_id: str = Field(default="airi-default", max_length=120)
    as_of: str | None = Field(default=None, max_length=64)


class RememberRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    namespace: str = Field(default="default", max_length=64)
    session_id: str = Field(default="airi-default", max_length=120)
    await_extraction: bool = True


class ConsolidateRequest(BaseModel):
    namespace: str = Field(default="default", max_length=64)


class ForgetRequest(BaseModel):
    target: str = Field(min_length=1, max_length=400)
    namespace: str = Field(default="default", max_length=64)


class ToggleStatusRequest(BaseModel):
    session_id: str = Field(default="airi-control", min_length=1, max_length=120)
    namespace: str = Field(default="default", max_length=64)


class CoreMemoryUpsertRequest(BaseModel):
    content: str = Field(min_length=1, max_length=800)
    namespace: str = Field(default="default", max_length=64)
    source_item_ids: list[str] = Field(default_factory=list)


class SummarizeRequest(BaseModel):
    namespace: str = Field(default="default", max_length=64)
    force: bool = False


class ImportMemoryRequest(BaseModel):
    payload: dict[str, Any]
    target_namespace: str | None = Field(default=None, max_length=64)


def resolve_namespace_and_model(
    requested_model: str,
    header_namespace: str | None,
    default_upstream_model: str,
) -> tuple[str, str]:
    """Parse AIRI model string ('hmph', 'hmph/yuki', 'hmph:yuki', or upstream model) & namespace header."""
    raw = (requested_model or "hmph").strip()
    inferred_ns: str | None = None
    upstream_model = default_upstream_model

    lower = raw.lower()
    if lower in {"hmph", "hmph-memory", "airi-memory"}:
        inferred_ns = "default"
    elif lower.startswith(("hmph/", "hmph:", "airi/")):
        sep = "/" if "/" in raw else ":"
        _, suffix = raw.split(sep, 1)
        # Allow optional explicit upstream model syntax: hmph/<ns>@<upstream_model>
        if "@" in suffix:
            ns_part, model_part = suffix.split("@", 1)
            inferred_ns = ns_part.strip() or "default"
            if model_part.strip():
                upstream_model = model_part.strip()
        else:
            inferred_ns = suffix.strip() or "default"
    else:
        upstream_model = raw

    ns = normalize_namespace(header_namespace or inferred_ns or "default")
    return ns, upstream_model


def _extract_choice_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    return str(content) if content else ""


def _extract_delta_text(chunk: dict[str, Any]) -> str:
    parts: list[str] = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            parts.append(str(content))
    return "".join(parts)


def create_airi_gateway_app(
    engine: LocalMemoryEngine,
    *,
    owns_engine: bool = False,
    gateway_api_key: str | None = None,
) -> FastAPI:
    """Create the headless FastAPI AIRI Memory Gateway (OpenAI-compatible + MCP HTTP + REST)."""
    mcp_server = LocalMCPServer(engine)
    expected_key = (
        gateway_api_key
        if gateway_api_key is not None
        else os.getenv("HMPH_GATEWAY_API_KEY", "")
    ).strip() or None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await engine.recover_pending_extractions()
        try:
            yield
        finally:
            if owns_engine:
                await engine.close()
            else:
                await engine.flush()

    app = FastAPI(
        title="Hmph Memory Gateway (AIRI & MCP)",
        version=__version__,
        description="Local SQLite memory gateway for Project AIRI and MCP agents.",
        lifespan=lifespan,
    )

    if expected_key:
        @app.middleware("http")
        async def _bearer_auth_guard(request: Request, call_next):
            if request.method == "OPTIONS" or request.url.path in {"/", "/v1/health"}:
                return await call_next(request)
            auth = request.headers.get("authorization", "").strip()
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if not token:
                token = request.headers.get("x-hmph-api-key", "").strip()
            if not token or not hmac.compare_digest(token, expected_key):
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Unauthorized: invalid or missing gateway API key"}},
                )
            return await call_next(request)

    # Enable permissive CORS so AIRI Web (airi.moeru.ai / localhost) and AIRI Desktop (Tauri)
    # can talk directly to http://127.0.0.1:8018/v1 without browser preflight errors.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Hmph-Namespace",
            "X-Hmph-Session-Id",
            "X-Hmph-Memory-Items",
            "X-Hmph-Consolidations",
            "X-Hmph-Retrieval-Ms",
        ],
    )

    @app.get("/")
    async def root():
        return {
            "service": "hmph-memory-gateway",
            "version": __version__,
            "mode": "headless-memory-component",
            "target_frontends": ["Project AIRI (Web & Desktop)", "MCP Agents"],
            "protocols": ["openai-compatible", "mcp-http", "memory-rest", "memory-events-sse"],
            "endpoints": {
                "openai_models": "/v1/models",
                "openai_chat_completions": "/v1/chat/completions",
                "mcp_jsonrpc": "/mcp",
                "health": "/v1/health",
                "memory_list": "/v1/memory",
                "memory_as_of": "/v1/memory/as-of",
                "memory_export": "/v1/memory/export",
                "memory_import": "/v1/memory/import",
                "memory_core": "/v1/memory/core",
                "memory_events_sse": "/v1/memory/events",
                "companion_proactive": "/v1/companion/proactive",
                "memory_graph": "/v1/memory/graph",
                "memory_recall": "/v1/memory/recall",
                "memory_remember": "/v1/memory/remember",
                "memory_extractions": "/v1/memory/extractions",
                "memory_summarize": "/v1/memory/summarize",
                "memory_consolidate": "/v1/memory/consolidate",
                "memory_forget": "/v1/memory/forget",
            },
        }

    @app.get("/v1/health")
    async def health():
        namespaces = engine.list_namespaces()
        return {
            "status": "ok",
            "version": __version__,
            "auth_enabled": bool(expected_key),
            "upstream_model": engine.llm.model_name,
            "embedding": engine.embedder.name,
            "db_path": str(engine.store.db_path.resolve())
            if str(engine.store.db_path) != ":memory:"
            else ":memory:",
            "timezone": engine.timezone_name,
            "namespaces": namespaces,
            "auto_consolidate_every": engine.auto_consolidate_every,
            "session_idle_minutes": engine.session_idle_minutes,
            "skipped_trivial_extractions": engine.skipped_trivial_extractions,
            "latency": engine.latency_stats(),
        }

    # ------------------------------------------------------------------
    # Protocol 1: OpenAI-Compatible Memory Proxy for AIRI (xsai)
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models():
        now_ts = int(time.time())
        model_ids = ["hmph", engine.llm.model_name]
        for ns in engine.list_namespaces():
            if ns != "default":
                model_ids.append(f"hmph/{ns}")
        unique_ids = list(dict.fromkeys(model_ids))
        return {
            "object": "list",
            "data": [
                {
                    "id": mid,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": "hmph-memory-gateway",
                }
                for mid in unique_ids
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionsRequest,
        x_hmph_namespace: str | None = Header(default=None, alias="X-Hmph-Namespace"),
        x_hmph_session_id: str | None = Header(default=None, alias="X-Hmph-Session-Id"),
        x_hmph_memory_mode: str | None = Header(default=None, alias="X-Hmph-Memory-Mode"),
    ):
        ns, upstream_model = resolve_namespace_and_model(
            req.model, x_hmph_namespace, engine.llm.model_name
        )
        explicit_sid = bool(
            (x_hmph_session_id and x_hmph_session_id.strip())
            or (req.user and req.user.strip())
        )
        base_sid = (x_hmph_session_id or req.user or f"airi-{ns}").strip() or f"airi-{ns}"
        session_id = engine.resolve_episode_session_id(
            base_sid, namespace=ns, explicit=explicit_sid
        )
        memory_mode = (x_hmph_memory_mode or "auto").strip().lower()
        extra_params = req.model_extra or {}

        turn_start = time.perf_counter()
        augmented_messages, evidence, user_msg, recent_before, ts = engine.prepare_gateway_turn(
            req.messages,
            namespace=ns,
            session_id=session_id,
            memory_mode=memory_mode,
        )

        resp_headers = {
            "X-Hmph-Namespace": ns,
            "X-Hmph-Session-Id": session_id,
            "X-Hmph-Memory-Items": str(len(evidence.used_item_ids) if evidence else 0),
            "X-Hmph-Consolidations": str(
                len(evidence.used_consolidation_ids) if evidence else 0
            ),
            "X-Hmph-Retrieval-Ms": str(evidence.retrieval_ms if evidence else 0.0),
        }

        if req.stream:
            async def sse_generator():
                collected: list[str] = []
                ttft_ms: float | None = None
                chunk_id = f"chatcmpl-hmph-{uuid4().hex[:12]}"
                created_ts = int(time.time())
                proxy_stream = getattr(engine.llm, "proxy_chat_completion_stream", None)

                try:
                    if callable(proxy_stream):
                        async for chunk in proxy_stream(
                            messages=augmented_messages,
                            model=upstream_model,
                            temperature=req.temperature,
                            max_tokens=req.max_tokens,
                            extra_params=extra_params,
                        ):
                            piece = _extract_delta_text(chunk)
                            if piece:
                                if ttft_ms is None:
                                    ttft_ms = round((time.perf_counter() - turn_start) * 1000.0, 1)
                                collected.append(piece)
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    else:
                        # Fallback for lightweight mock LLMs without proxy_chat_completion_stream.
                        reply_text, _ = await engine.llm.generate_reply(
                            user_message=engine.extract_latest_user_text(augmented_messages),
                            recent_messages=[],
                            narrated_context=evidence.narrated_context if evidence else "",
                            now_iso=ts,
                            timezone_name=engine.timezone_name,
                        )
                        ttft_ms = round((time.perf_counter() - turn_start) * 1000.0, 1)
                        collected.append(reply_text)
                        synthetic = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": reply_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(synthetic, ensure_ascii=False)}\n\n"
                        finish_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": req.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    total_ms = round((time.perf_counter() - turn_start) * 1000.0, 1)
                    engine.finalize_gateway_turn(
                        user_msg=user_msg,
                        assistant_text="".join(collected),
                        evidence=evidence,
                        recent_before=recent_before,
                        namespace=ns,
                        session_id=session_id,
                        memory_mode=memory_mode,
                        ttft_ms=ttft_ms if ttft_ms is not None else total_ms,
                        total_ms=total_ms,
                    )

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={
                    **resp_headers,
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming OpenAI chat.completion
        proxy_fn = getattr(engine.llm, "proxy_chat_completion", None)
        if callable(proxy_fn):
            payload = await proxy_fn(
                messages=augmented_messages,
                model=upstream_model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                extra_params=extra_params,
            )
            assistant_text = _extract_choice_text(payload)
        else:
            assistant_text, usage = await engine.llm.generate_reply(
                user_message=engine.extract_latest_user_text(augmented_messages),
                recent_messages=[],
                narrated_context=evidence.narrated_context if evidence else "",
                now_iso=ts,
                timezone_name=engine.timezone_name,
            )
            payload = {
                "id": f"chatcmpl-hmph-{uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": assistant_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        total_ms = round((time.perf_counter() - turn_start) * 1000.0, 1)
        engine.finalize_gateway_turn(
            user_msg=user_msg,
            assistant_text=assistant_text,
            evidence=evidence,
            recent_before=recent_before,
            namespace=ns,
            session_id=session_id,
            memory_mode=memory_mode,
            ttft_ms=total_ms,
            total_ms=total_ms,
        )
        return Response(
            content=json.dumps(payload, ensure_ascii=False),
            media_type="application/json",
            headers=resp_headers,
        )

    # ------------------------------------------------------------------
    # Protocol 2: MCP Streamable HTTP JSON-RPC 2.0
    # ------------------------------------------------------------------

    @app.post("/mcp")
    async def mcp_http(request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON-RPC payload: {exc}") from None
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON-RPC payload must be an object")
        resp = await mcp_server.handle_request(body)
        if resp is None:
            return Response(status_code=204)
        return resp

    # ------------------------------------------------------------------
    # Protocol 3: Headless Memory REST API
    # ------------------------------------------------------------------

    @app.get("/v1/memory/namespaces")
    async def list_namespaces():
        return {"namespaces": engine.list_namespaces()}

    @app.get("/v1/memory")
    async def get_memory(namespace: str = "default"):
        ns = normalize_namespace(namespace)
        items = engine.list_items(namespace=ns)
        ambiguities = engine.list_open_ambiguities(namespace=ns)
        consolidations = engine.list_consolidations(namespace=ns)
        core_blocks = engine.get_core_memory_blocks(namespace=ns)
        summaries = engine.list_session_summaries(namespace=ns)
        item_map = {it.id: it.title for it in items}
        return {
            "namespace": ns,
            "core_blocks": [b.model_dump(mode="json") for b in core_blocks],
            "active": [it.model_dump(mode="json") for it in items if it.status == "active"],
            "paused": [it.model_dump(mode="json") for it in items if it.status == "paused"],
            "closed": [
                it.model_dump(mode="json")
                for it in items
                if it.status not in {"active", "paused"}
            ],
            "consolidations": [c.model_dump(mode="json") for c in consolidations],
            "session_summary_count": len(summaries),
            "ambiguities": [
                {
                    **amb.model_dump(mode="json"),
                    "candidate_titles": [
                        item_map.get(cid, cid[:8]) for cid in amb.candidate_item_ids
                    ],
                }
                for amb in ambiguities
            ],
        }

    @app.get("/v1/memory/core")
    async def get_core_memory(namespace: str = "default"):
        ns = normalize_namespace(namespace)
        return {
            "namespace": ns,
            "blocks": [b.model_dump(mode="json") for b in engine.get_core_memory_blocks(namespace=ns)],
        }

    @app.put("/v1/memory/core/{block_key}")
    async def put_core_memory(block_key: str, req: CoreMemoryUpsertRequest):
        blk = engine.set_core_memory_block(
            block_key,
            req.content,
            namespace=req.namespace,
            source_item_ids=req.source_item_ids,
        )
        return {"block": blk.model_dump(mode="json")}

    @app.delete("/v1/memory/core/{block_key}")
    async def delete_core_memory(block_key: str, namespace: str = "default"):
        deleted = engine.delete_core_memory_block(block_key, namespace=namespace)
        return {"deleted": deleted, "block_key": block_key, "namespace": normalize_namespace(namespace)}

    @app.get("/v1/companion/proactive")
    async def companion_proactive(namespace: str = "default", cooldown_hours: float = 12.0):
        suggestion = engine.get_proactive_suggestion(
            namespace=namespace, cooldown_hours=max(0.0, cooldown_hours)
        )
        return {"suggestion": suggestion.model_dump(mode="json")}

    @app.post("/v1/memory/summarize")
    async def summarize_sessions(req: SummarizeRequest | None = None):
        ns = req.namespace if req else "default"
        force = req.force if req else False
        return await engine.backfill_session_summaries(namespace=ns, force=force)

    @app.get("/v1/memory/events")
    async def memory_events_sse(
        request: Request,
        namespace: str = "default",
        max_events: int | None = None,
    ):
        """Real-time SSE memory event stream for AIRI Desktop plugins / Live2D triggers."""
        ns = "*" if namespace.strip() == "*" else normalize_namespace(namespace)
        queue = engine.subscribe_events(namespace=ns)

        async def event_stream():
            emitted = 0
            try:
                yield f"event: connected\ndata: {json.dumps({'namespace': ns, 'version': __version__})}\n\n"
                while True:
                    if max_events is not None and emitted >= max_events:
                        break
                    if await request.is_disconnected():
                        break
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"event: {evt.event_type}\ndata: {json.dumps(evt.model_dump(mode='json'), ensure_ascii=False)}\n\n"
                        emitted += 1
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                engine.unsubscribe_events(queue, namespace=ns)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/v1/memory/as-of")
    async def get_memory_as_of(
        as_of: str,
        namespace: str = "default",
        include_closed: bool = False,
    ):
        ns = normalize_namespace(namespace)
        items = engine.get_items_as_of(as_of, namespace=ns, include_closed=include_closed)
        return {
            "namespace": ns,
            "as_of": as_of,
            "items": [it.model_dump(mode="json") for it in items],
        }

    @app.get("/v1/memory/export")
    async def export_memory(namespace: str = "default", format: str = "json"):
        ns = normalize_namespace(namespace)
        if format.strip().lower() in {"md", "markdown"}:
            md = engine.export_namespace(namespace=ns, fmt="markdown")
            return PlainTextResponse(
                content=str(md),
                media_type="text/markdown; charset=utf-8",
            )
        return engine.export_namespace(namespace=ns, fmt="json")

    @app.post("/v1/memory/import")
    async def import_memory(req: ImportMemoryRequest):
        stats = engine.import_namespace(
            req.payload, target_namespace=req.target_namespace
        )
        return {"imported": stats}

    @app.get("/v1/memory/graph")
    async def get_memory_graph(namespace: str = "default"):
        return engine.export_graph_data(namespace=namespace)

    @app.post("/v1/memory/recall")
    async def recall_memory(req: RecallRequest):
        ev = engine.recall_context(
            req.query,
            namespace=req.namespace,
            session_id=req.session_id,
            as_of=req.as_of,
        )
        return {"evidence": ev.model_dump(mode="json")}

    @app.post("/v1/memory/remember")
    async def remember_memory(req: RememberRequest):
        msg = await engine.ingest_user_message(
            req.session_id,
            req.text,
            namespace=req.namespace,
            await_extraction=req.await_extraction,
        )
        return {
            "message": msg.model_dump(mode="json"),
            "item_count": len(engine.list_items(namespace=req.namespace)),
        }

    @app.get("/v1/memory/extractions")
    async def list_extractions(namespace: str = "default"):
        return {
            "quarantined": engine.store.list_quarantined_messages(namespace=namespace),
        }

    @app.post("/v1/memory/extractions/{message_id}/retry")
    async def retry_extraction(message_id: str, namespace: str = "default"):
        message = await engine.retry_quarantined_extraction(message_id, namespace=namespace)
        if message is None:
            raise HTTPException(status_code=404, detail="Quarantined message not found")
        if message.extraction_failure == "stale_source":
            raise HTTPException(status_code=409, detail="A newer source already updated this item")
        return {"message": message}

    @app.post("/v1/memory/consolidate")
    async def consolidate_memory(req: ConsolidateRequest | None = None):
        ns = req.namespace if req else "default"
        report = await engine.consolidate(namespace=ns)
        return {"report": report.model_dump(mode="json")}

    @app.post("/v1/memory/forget")
    async def forget_memory(req: ForgetRequest):
        receipt = await engine.forget(req.target, namespace=req.namespace)
        return {"receipt": receipt.model_dump(mode="json")}

    @app.post("/v1/memory/items/{item_id}/toggle")
    async def toggle_item(item_id: str, req: ToggleStatusRequest):
        updated = await engine.toggle_item_pause(
            item_id, session_id=req.session_id, namespace=req.namespace
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Memory item not found in namespace")
        return {"item": updated.model_dump(mode="json")}

    return app
