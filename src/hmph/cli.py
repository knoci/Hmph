"""Command-line interface for Hmph headless memory gateway, MCP server, and local CLI chat."""

from __future__ import annotations

import argparse
import asyncio
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
        help="Trigger background Sleep Consolidation every N user turns (0 disables; default: 12).",
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


if __name__ == "__main__":
    main()
