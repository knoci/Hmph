import argparse
import asyncio
import secrets
from pathlib import Path

from hmph.config import Settings
from hmph.conversation import Conversation
from hmph.database import Database


def initialize():
    path = Path(".env")
    # Exclusive create: running init again must never rotate existing credentials.
    with path.open("x", encoding="utf-8") as output:
        output.write(
            "HMPH_DATABASE_URL=postgresql+asyncpg://hmph:hmph_local_only@127.0.0.1:54329/hmph\n"
            f"HMPH_API_TOKEN={secrets.token_urlsafe(32)}\n"
            f"HMPH_REQUEST_DIGEST_KEY={secrets.token_urlsafe(32)}\n"
            "HMPH_MODEL_MODE=fake\n"
        )
    print("Created .env with local credentials. Keep this file private.")


async def run_database_command(name):
    settings = Settings()
    database = Database(settings.database_url.get_secret_value(), schema=settings.database_schema)
    try:
        if name == "migrate":
            await database.migrate()
            print("Database migrated.")
        else:
            from hmph.runtime import build_worker

            conversation = Conversation(
                database,
                digest_key=settings.request_digest_key.get_secret_value().encode(),
                turn_timeout=settings.turn_timeout_seconds,
            )
            await conversation.bootstrap()
            print(f"Worker running in {settings.model_mode} mode (local testing).")
            await build_worker(conversation, settings).run_forever()
    finally:
        await database.close()


def main():
    parser = argparse.ArgumentParser(prog="hmph")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create local configuration without overwriting it")
    commands.add_parser("migrate", help="Apply versioned PostgreSQL migrations")
    commands.add_parser("worker", help="Run the durable generation worker")
    evaluate = commands.add_parser("eval", help="Audit evaluation input isolation; no model calls")
    evaluate.add_argument("--suite", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    serve = commands.add_parser("serve", help="Run the local authenticated HTTP API")
    serve.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "::1"])
    serve.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    if args.command == "init":
        try:
            initialize()
        except FileExistsError:
            parser.exit(1, ".env already exists; nothing changed.\n")
    elif args.command == "eval":
        import json

        from hmph.evaluation import audit_suite

        report = audit_suite(args.suite)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Input audit passed for {report['case_count']} cases. Quality evaluation: NOT RUN.")
    elif args.command == "serve":
        import uvicorn

        from hmph.api import create_app

        uvicorn.run(
            create_app(), host=args.host, port=args.port, access_log=False, proxy_headers=False
        )
    else:
        try:
            asyncio.run(run_database_command(args.command))
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
