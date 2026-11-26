"""Command-line interface for Hmph headless memory gateway, MCP server, local chat, and export/import."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from hmph import __version__

DEFAULT_DB_PATH = "data/hmph.db"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hmph",
        description="Hmph — Headless single-file SQLite long-term memory for Project AIRI & MCP.",
    )
    parser.add_argument("--version", action="version", version=f"hmph {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    gateway = subparsers.add_parser(
        "airi-gateway",
        aliases=["local-gateway"],
        help="Run the headless AIRI Memory Gateway (OpenAI-compatible + MCP HTTP + REST + SSE).",
    )
    gateway.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to the single-file SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    gateway.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    gateway.add_argument("--port", type=int, default=8018, help="Bind port (default: 8018).")
    gateway.add_argument(
        "--auto-consolidate-every",
        type=int,
        default=12,
        help="Trigger background Sleep Consolidation every N mutations (0 disables; default: 12).",
    )

    mcp = subparsers.add_parser(
        "local-mcp",
        help="Run the zero-dependency MCP JSON-RPC 2.0 server over stdio.",
    )
    mcp.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to the single-file SQLite database (default: {DEFAULT_DB_PATH}).",
    )

    chat = subparsers.add_parser(
        "local-chat",
        help="Start an interactive CLI chat session backed by local SQLite memory.",
    )
    chat.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to the single-file SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    chat.add_argument("--session-id", default=None, help="Optional explicit session ID.")

    exp = subparsers.add_parser(
        "export",
        help="Export a character namespace to portable JSON or human-readable Markdown.",
    )
    exp.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path.")
    exp.add_argument("--namespace", default="default", help="Character namespace (default: default).")
    exp.add_argument(
        "--format",
        choices=["json", "markdown", "md"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    exp.add_argument("--output", default=None, help="Optional output file path (stdout if omitted).")

    imp = subparsers.add_parser(
        "import",
        help="Import a portable JSON memory snapshot into a character namespace.",
    )
    imp.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite DB path.")
    imp.add_argument("--input", required=True, help="Path to exported JSON snapshot file.")
    imp.add_argument(
        "--namespace",
        default=None,
        help="Optional target namespace override (defaults to snapshot namespace).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in {"airi-gateway", "local-gateway"}:
        from hmph.local_memory.cli import run_airi_gateway

        run_airi_gateway(
            Path(args.db),
            host=args.host,
            port=args.port,
            auto_consolidate_every=args.auto_consolidate_every,
        )
        return

    if args.command == "local-mcp":
        from hmph.local_memory.mcp_server import run_mcp_stdio

        asyncio.run(run_mcp_stdio(Path(args.db)))
        return

    if args.command == "local-chat":
        from hmph.local_memory.cli import run_local_chat

        asyncio.run(run_local_chat(Path(args.db), session_id=args.session_id))
        return

    if args.command == "export":
        from hmph.local_memory.embeddings import build_default_embedder
        from hmph.local_memory.store import LocalMemoryStore

        store = LocalMemoryStore(Path(args.db), embedder=build_default_embedder())
        try:
            if args.format in {"markdown", "md"}:
                rendered = store.export_namespace_markdown(namespace=args.namespace)
            else:
                rendered = json.dumps(
                    store.export_namespace_data(namespace=args.namespace),
                    ensure_ascii=False,
                    indent=2,
                ) + "\n"
            if args.output:
                out_path = Path(args.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(rendered, encoding="utf-8")
                print(f"Exported namespace '{args.namespace}' -> {out_path}")
            else:
                sys.stdout.write(rendered)
        finally:
            store.close()
        return

    if args.command == "import":
        from hmph.local_memory.embeddings import build_default_embedder
        from hmph.local_memory.store import LocalMemoryStore

        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        store = LocalMemoryStore(Path(args.db), embedder=build_default_embedder())
        try:
            stats = store.import_namespace_data(payload, target_namespace=args.namespace)
            print(json.dumps(stats, ensure_ascii=False))
        finally:
            store.close()
        return


if __name__ == "__main__":
    main()
