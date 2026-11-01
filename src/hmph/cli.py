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


async def run_database_command(name, *, max_calls=64, once=False):
    settings = Settings()
    database = Database(settings.database_url.get_secret_value(), schema=settings.database_schema)
    try:
        if name == "migrate":
            await database.migrate()
            print("Database migrated.")
        else:
            from hmph.runtime import build_memory_worker, build_state_worker, build_worker

            conversation = Conversation(
                database,
                digest_key=settings.request_digest_key.get_secret_value().encode(),
                turn_timeout=settings.turn_timeout_seconds,
                source_timezone=settings.source_timezone,
            )
            await conversation.bootstrap()
            if name == "framework-worker":
                from hmph.framework_worker import FrameworkWorker
                from hmph.mem0_backend import Mem0Backend
                from hmph.memory_builds import MemoryBuilds

                if settings.deepseek_api_key is None:
                    raise ValueError("Framework worker requires a local DeepSeek key")
                backend = Mem0Backend(
                    settings.deepseek_api_key,
                    database_url=settings.database_url.get_secret_value(),
                    max_calls=max_calls,
                )
                worker = FrameworkWorker(
                    MemoryBuilds(database, digest_key=conversation.digest_key), backend
                )
                print("Mem0 shadow worker running; requested builds may call DeepSeek.")
                if once:
                    await worker.run_once()
                else:
                    await worker.run_forever()
                return
            print(f"Worker running in {settings.model_mode} mode (local testing).")
            factory = {
                "memory-worker": build_memory_worker,
                "state-worker": build_state_worker,
                "worker": build_worker,
            }[name]
            worker = factory(conversation, settings)
            await worker.run_forever()
    finally:
        await database.close()


async def run_evaluation(args):
    import json
    import time

    from hmph.deepseek import DeepSeekModel
    from hmph.eval_runner import run_suite, save_report

    settings = Settings(model_mode="fake") if args.retrieval_only else Settings()
    if not args.retrieval_only and settings.deepseek_api_key is None:
        raise ValueError("E1 requires a local DeepSeek key")
    model = (
        None
        if args.retrieval_only
        else DeepSeekModel(settings.deepseek_api_key, model=settings.deepseek_model)
    )
    semantic_ranker = None
    if "hybrid_raw" in args.profiles:
        from hmph.semantic import LocalSemanticRanker

        semantic_ranker = await asyncio.to_thread(LocalSemanticRanker)
    metadata = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    await asyncio.to_thread(args.output.parent.mkdir, parents=True, exist_ok=True)

    last_checkpoint = 0

    async def checkpoint(report):
        nonlocal last_checkpoint
        if args.retrieval_only and time.monotonic() - last_checkpoint < 1:
            return
        await asyncio.to_thread(save_report, args.output, report)
        last_checkpoint = time.monotonic()

    report = await run_suite(
        args.suite,
        model.read_evaluation if model else None,
        judge=model.judge_evaluation if model and args.judge else None,
        profiles=tuple(args.profiles),
        limit=args.limit,
        seed=args.seed,
        budget_bytes=args.budget_bytes,
        model=settings.deepseek_model if model else "not_run",
        dataset_condition=args.condition,
        dataset_metadata=metadata,
        progress=checkpoint,
        semantic_ranker=semantic_ranker,
        retrieval_only=args.retrieval_only,
        provider_config={
            "reader_prompt": "deepseek-e1-reader-v2",
            "judge_prompt": "hmph-binary-judge-v1" if args.judge else None,
            "reader_max_output_tokens": 256,
            "judge_max_output_tokens": 64,
            "thinking": "disabled",
        },
    )
    await asyncio.to_thread(save_report, args.output, report)
    print(
        f"E1 completed {report['case_count']} cases; "
        f"{report['inference_requests']} attempted calls. "
        "Official score: NO. Frontier status: NOT EVALUATED."
    )


def main():
    parser = argparse.ArgumentParser(prog="hmph")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create local configuration without overwriting it")
    commands.add_parser("migrate", help="Apply versioned PostgreSQL migrations")
    commands.add_parser("worker", help="Run the durable generation worker")
    commands.add_parser("memory-worker", help="Run sourced memory extraction independently")
    commands.add_parser("state-worker", help="Maintain grounded short-term state and watermarks")
    framework = commands.add_parser(
        "framework-worker", help="Run opt-in persistent Mem0 shadow builds"
    )
    framework.add_argument(
        "--max-calls",
        type=int,
        choices=range(1, 129),
        default=64,
        metavar="1..128",
        help="Provider attempt budget per build attempt",
    )
    framework.add_argument(
        "--once", action="store_true", help="Process at most one build or cleanup"
    )
    evaluate = commands.add_parser("eval", help="Audit evaluation input isolation; no model calls")
    evaluate.add_argument("--suite", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser(
        "eval-run", help="Run opt-in DeepSeek memory QA baselines; paid calls"
    )
    infer.add_argument("--suite", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--manifest", type=Path)
    infer.add_argument(
        "--condition", required=True, help="Disclose history condition, e.g. cleaned-S"
    )
    infer.add_argument(
        "--profiles",
        nargs="+",
        choices=["recent_window", "lexical_raw", "source_aware_raw", "hybrid_raw", "full_history"],
        default=["recent_window", "lexical_raw", "full_history"],
    )
    infer.add_argument("--limit", type=int, default=5)
    infer.add_argument("--seed", type=int, default=17)
    infer.add_argument("--budget-bytes", type=int, default=12000)
    infer.add_argument("--judge", action="store_true", help="Use separate DeepSeek grading calls")
    infer.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate source retrieval without provider calls",
    )
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
    elif args.command == "eval-run":
        asyncio.run(run_evaluation(args))
    else:
        try:
            asyncio.run(
                run_database_command(
                    args.command,
                    max_calls=getattr(args, "max_calls", 64),
                    once=getattr(args, "once", False),
                )
            )
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
