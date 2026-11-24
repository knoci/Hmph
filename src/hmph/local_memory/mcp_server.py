"""Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 server for Hmph local memory.

Shared by both stdio transport (`hmph local-mcp`) and Streamable HTTP (`POST /mcp` on `hmph airi-gateway`).
Allows AIRI Desktop (Tauri MCP), Claude Desktop, Cursor, and any MCP agent to mount
Hmph's multi-namespace, evolving SQLite memory engine directly via 8 standard tools:
  - hmph_remember
  - hmph_recall
  - hmph_list_memories
  - hmph_forget
  - hmph_consolidate
  - hmph_graph
  - hmph_toggle_pause
  - hmph_core_memory
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from hmph.local_memory.engine import LocalMemoryEngine

PROTOCOL_VERSION = "2024-11-05"

_NS_PROP = {
    "type": "string",
    "description": "Optional character/agent memory namespace for strict isolation (default: default).",
}

TOOLS_SPEC = [
    {
        "name": "hmph_remember",
        "description": (
            "Store a user utterance into Hmph's single-file SQLite memory ledger and run "
            "structured state-evolution extraction (create / reschedule / update / pause / resume / retract)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Verbatim user statement to remember."},
                "session_id": {
                    "type": "string",
                    "description": "Optional conversation session ID (default: mcp-default).",
                },
                "namespace": _NS_PROP,
            },
            "required": ["text"],
        },
    },
    {
        "name": "hmph_recall",
        "description": (
            "Recall relevant active/paused states, vector-indexed consolidated user insights, "
            "full version evolution chains, and verbatim provenance excerpts using ONNX + FTS5 + Bipartite PPR."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question or topic to recall memory for."},
                "session_id": {
                    "type": "string",
                    "description": "Optional current session ID (default: mcp-default).",
                },
                "namespace": _NS_PROP,
            },
            "required": ["query"],
        },
    },
    {
        "name": "hmph_list_memories",
        "description": "List tracked evolving memory items, consolidated insights, and open ambiguities in a namespace.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "paused", "completed", "cancelled", "retracted"],
                    "description": "Optional status filter.",
                },
                "namespace": _NS_PROP,
            },
        },
    },
    {
        "name": "hmph_forget",
        "description": (
            "Atomically cascade-forget a memory item (by ID or keyword) or source message ID inside a namespace, "
            "removing all derived versions, consolidations, edges, vectors, and linked replies."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Item ID, message ID, or keyword identifying the memory to forget.",
                },
                "namespace": _NS_PROP,
            },
            "required": ["target"],
        },
    },
    {
        "name": "hmph_consolidate",
        "description": (
            "Run sleep consolidation for a namespace: merge semantically near-duplicate active items, "
            "synthesize provenance-backed vector-indexed long-term user insights, and decay stale episode salience."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"namespace": _NS_PROP},
        },
    },
    {
        "name": "hmph_graph",
        "description": (
            "Export the namespace's Memory Evolution Relationship Graph (items, shared entities, "
            "consolidated insights, and DERIVED_FROM / MENTIONS / RELATED_ENTITY edges) as structured JSON."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"namespace": _NS_PROP},
        },
    },
    {
        "name": "hmph_toggle_pause",
        "description": (
            "Pause an active tracked plan/item (suppressing proactive nagging) or resume a paused item "
            "while preserving its last active schedule."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_id": {"type": "string", "description": "Memory item UUID to pause or resume."},
                "session_id": {"type": "string", "description": "Optional session ID."},
                "namespace": _NS_PROP,
            },
            "required": ["item_id"],
        },
    },
    {
        "name": "hmph_core_memory",
        "description": (
            "Read, update, or delete Letta-style always-in-context Core Memory blocks "
            "(e.g., user_profile, relationship_style, boundaries) for a namespace."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "delete"],
                    "description": "Operation to perform (default: get).",
                },
                "block_key": {
                    "type": "string",
                    "description": "Core block key (required for set/delete, e.g., user_profile).",
                },
                "content": {
                    "type": "string",
                    "description": "Block content (required for set).",
                },
                "namespace": _NS_PROP,
            },
        },
    },
]


class LocalMCPServer:
    """Handles MCP JSON-RPC requests against a LocalMemoryEngine instance."""

    def __init__(self, engine: LocalMemoryEngine):
        self.engine = engine

    async def handle_request(self, req: dict[str, Any]) -> dict[str, Any] | None:
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params") or {}

        if method == "notifications/initialized":
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "hmph-local-memory", "version": "0.5.0"},
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SPEC}}

        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                output = await self._call_tool(str(name), args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(output, ensure_ascii=False, indent=2)}
                        ]
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": True,
                        "content": [{"type": "text", "text": f"Tool error: {exc}"}],
                    },
                }

        if req_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    async def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        ns = str(args.get("namespace") or "default").strip() or "default"

        if name == "hmph_remember":
            text = str(args.get("text", "")).strip()
            if not text:
                raise ValueError("text is required")
            sid = str(args.get("session_id") or "mcp-default")
            msg = await self.engine.ingest_user_message(
                sid, text, namespace=ns, await_extraction=True
            )
            return {
                "message_id": msg.id,
                "namespace": msg.namespace,
                "session_id": msg.session_id,
                "items_total": len(self.engine.list_items(namespace=ns)),
                "open_ambiguities": len(self.engine.list_open_ambiguities(namespace=ns)),
            }

        if name == "hmph_recall":
            query = str(args.get("query", "")).strip()
            if not query:
                raise ValueError("query is required")
            sid = str(args.get("session_id") or "mcp-default")
            evidence = self.engine.recall_context(query, namespace=ns, session_id=sid)
            return evidence.model_dump(mode="json")

        if name == "hmph_list_memories":
            status = args.get("status")
            items = (
                self.engine.list_items(namespace=ns, status=status)
                if status
                else self.engine.list_items(namespace=ns)
            )
            return {
                "namespace": ns,
                "namespaces": self.engine.list_namespaces(),
                "items": [it.model_dump(mode="json") for it in items],
                "consolidations": [
                    c.model_dump(mode="json")
                    for c in self.engine.list_consolidations(namespace=ns)
                ],
                "open_ambiguities": [
                    a.model_dump(mode="json")
                    for a in self.engine.list_open_ambiguities(namespace=ns)
                ],
            }

        if name == "hmph_forget":
            target = str(args.get("target", "")).strip()
            if not target:
                raise ValueError("target is required")
            receipt = await self.engine.forget(target, namespace=ns)
            return receipt.model_dump(mode="json")

        if name == "hmph_consolidate":
            report = await self.engine.consolidate(namespace=ns)
            return report.model_dump(mode="json")

        if name == "hmph_graph":
            return self.engine.export_graph_data(namespace=ns)

        if name == "hmph_toggle_pause":
            item_id = str(args.get("item_id", "")).strip()
            if not item_id:
                raise ValueError("item_id is required")
            sid = str(args.get("session_id") or "mcp-control")
            updated = await self.engine.toggle_item_pause(
                item_id, session_id=sid, namespace=ns
            )
            if updated is None:
                raise ValueError(f"Memory item not found in namespace '{ns}': {item_id}")
            return {"item": updated.model_dump(mode="json")}

        if name == "hmph_core_memory":
            action = str(args.get("action") or "get").strip().lower()
            if action == "get":
                return {
                    "namespace": ns,
                    "blocks": [
                        b.model_dump(mode="json")
                        for b in self.engine.get_core_memory_blocks(namespace=ns)
                    ],
                }
            block_key = str(args.get("block_key") or "").strip()
            if not block_key:
                raise ValueError("block_key is required for set/delete")
            if action == "set":
                content = str(args.get("content") or "").strip()
                if not content:
                    raise ValueError("content is required for set")
                blk = self.engine.set_core_memory_block(block_key, content, namespace=ns)
                return {"block": blk.model_dump(mode="json")}
            if action == "delete":
                deleted = self.engine.delete_core_memory_block(block_key, namespace=ns)
                return {"deleted": deleted, "block_key": block_key, "namespace": ns}
            raise ValueError(f"Unsupported core memory action: {action}")

        raise ValueError(f"Unknown tool: {name}")


async def run_mcp_stdio(db_path: Path | str = Path("artifacts/local-memory/hmph.db")) -> None:
    """Run the MCP JSON-RPC server over stdin/stdout until EOF."""
    engine = LocalMemoryEngine.from_env(db_path)
    server = LocalMCPServer(engine)
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = await server.handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
    finally:
        await engine.close()
