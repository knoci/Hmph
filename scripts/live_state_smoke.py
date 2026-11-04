"""Explicit, frozen state/recall or companion demo in a temporary database schema."""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from hmph.config import Settings
from hmph.conversation import Conversation
from hmph.database import Database

ROOT = Path(__file__).resolve().parents[1]


async def shared_threads_demo(directory, execute):
    """New C2-only demo; old state smoke is never implicitly rerun."""
    import hashlib

    from hmph.deepseek import DeepSeekModel
    from hmph.eval_budget import RunBudget
    from hmph.product_eval import ProductTransport
    from hmph.shared_threads import SharedThreads
    from hmph.timem_reference import local_test_key
    from hmph.worker import Worker

    root = ROOT
    files = [*root.glob("src/hmph/**/*.py"), Path(__file__)]
    hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    plan = {
        "suite": "shared-thread-product-demo-v1",
        "code_hashes": hashes,
        "model": "deepseek-flash",
        "topic": "每周一起聊聊练琴进展",
        "human_review": "not_run",
        "utterances": [
            "我回来了，今天想和你聊聊练琴。",
            "今天下班了，有点累。",
            "我回来了，继续聊聊练琴的事吧。",
            "晚安，今天先到这里。",
        ],
        "limits": {"max_calls": 8, "max_input_bytes": 128 * 1024, "per_case_calls": 8},
        "stopping": "first technical failure, STOP or budget exhaustion; no paid retry",
    }
    if not execute:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Prepared new C2 demo; real calls=0")
        return
    frozen = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if frozen != plan or (directory / "report.json").exists():
        raise ValueError("Frozen demo changed or already started")
    settings = Settings()
    key = local_test_key(ROOT)
    budget = RunBudget(directory / "calls.sqlite", **plan["limits"])
    transports = {
        stage: ProductTransport(
            key,
            budget=budget.bind("p5-dev-001", "lexical", stage),
            stop_file=directory / "STOP",
        )
        for stage in ("generation", "guard")
    }
    models = {
        stage: DeepSeekModel(key, transport=transport) for stage, transport in transports.items()
    }
    schema = "live_c2_" + uuid4().hex
    admin = create_async_engine(settings.database_url.get_secret_value())
    db = Database(settings.database_url.get_secret_value(), schema=schema)
    report = {"status": "running", "rows": [], "human_review": "not_run"}
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        await db.migrate()
        conversation = Conversation(db, digest_key=b"synthetic-c2-only-request-digest")
        identity = await conversation.bootstrap()
        chat = await conversation.create_chat(identity.owner_id, identity.character_id, "first")
        controls = SharedThreads(conversation)
        thread = await controls.write(
            identity.owner_id, chat.id, "create", topic=plan["topic"], model_mode="deepseek"
        )
        chat = await conversation.create_chat(identity.owner_id, identity.character_id, "second")
        worker = Worker(
            conversation,
            models["generation"],
            model_mode="deepseek",
            model_name="deepseek-flash",
            reviewer=models["guard"].review,
        )
        from uuid import UUID

        for index, utterance in enumerate(plan["utterances"]):
            if index in (1, 2):
                thread = await controls.write(
                    identity.owner_id,
                    chat.id,
                    f"control-{index}",
                    thread_id=UUID(thread["id"]),
                    expected_revision=thread["revision"],
                    status="paused" if index == 1 else "open",
                    model_mode="deepseek",
                )
            if (directory / "STOP").exists():
                raise RuntimeError("Stopped")
            turn = await conversation.accept(
                identity.owner_id, chat.id, f"demo-{index}", utterance, model_mode="deepseek"
            )
            await worker.run_once()
            saved = await conversation.get_turn(identity.owner_id, turn.id)
            messages = await conversation.messages(identity.owner_id, chat.id)
            reply = next(
                (m.content for m in messages if m.turn_id == turn.id and m.role == "assistant"), ""
            )
            explanation = await conversation.companion_explanation(identity.owner_id, turn.id)
            report["rows"].append(
                {
                    "utterance": utterance,
                    "reply": reply,
                    "status": saved.status,
                    "error_code": saved.error_code,
                    "usage": saved.usage,
                    "plan": explanation["plan"],
                }
            )
            (directory / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if saved.status != "completed":
                raise RuntimeError("Technical turn failure")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
    finally:
        for transport in transports.values():
            await transport.shutdown()
        await db.close()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
        report["temporary_schema_removed"] = True
        report["budget"] = budget.snapshot()
        (directory / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "turns": len(report["rows"]),
                "calls": len(report["budget"]["calls"]),
            }
        )
    )


async def recall_demo(directory, execute, *, selected_case=None, selected_profile=None):
    """New P4-c product development; no old fixtures or paid replay."""
    import hashlib
    import sys
    import time
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "scripts"))
    from product_run import CapturingConversation, Services, evaluation_error_code, save
    from pydantic import RootModel
    from sqlalchemy import select

    from hmph.dialogue import Dialogue
    from hmph.dialogue_worker import StateWorker
    from hmph.eval_budget import RunBudget
    from hmph.models import Turn
    from hmph.product_eval import (
        ANSWER_PROMPT,
        GROUND_PROMPT,
        AnswerScore,
        GroundingScore,
        RunControl,
        actual_context,
        grade_product,
    )
    from hmph.selection_eval import Query, Source
    from hmph.timem_reference import local_test_key
    from hmph.worker import Worker

    cases = [
        {
            "history": [],
            "seed": "我周六要去北岸工作室面试。",
            "lag": 0,
            "fillers": [],
            "query": "那家后来有什么安排？",
            "answer": "周六去北岸工作室面试",
            "aliases": ["星期六去北岸工作室面试"],
            "required": ["s001"],
        },
        {
            "history": ["我姐姐叫露西。", "露西在里昂的一家银行上班。"],
            "seed": "周末我想整理书架。",
            "lag": 1,
            "fillers": ["今天午饭吃了面条。"],
            "query": "我姐姐在哪个城市工作？",
            "answer": "里昂",
            "aliases": ["Lyon"],
            "required": ["s001", "s002"],
        },
        {
            "history": [],
            "seed": "我周六要去北岸工作室面试。",
            "lag": 5,
            "fillers": [
                "今天把桌子整理好了。",
                "窗外有点风。",
                "刚听完一首轻音乐。",
                "晚饭准备煮面。",
                "今天穿了一件蓝色外套。",
            ],
            "query": "原来北岸工作室的面试取消，改成明天去远山公司。我现在要去哪家公司？",
            "answer": "远山公司",
            "aliases": [],
            "required": ["s007"],
        },
    ]
    files = [Path(__file__), ROOT / "scripts/product_run.py", *ROOT.glob("src/hmph/**/*.py")]
    plan = {
        "purpose": "P4-c new product recall/state development, not formal or human acceptance",
        "cases": cases,
        "profiles": [selected_profile] if selected_profile else ["lexical", "focused"],
        "selected_cases": [selected_case] if selected_case else [1, 2, 3],
        "model": "deepseek-flash",
        "hashes": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
        "limits": {
            "max_calls": 6 * (1 if selected_case else 3) * (1 if selected_profile else 2),
            "max_input_bytes": 2 * 1024 * 1024,
            "per_case_calls": 6 * (1 if selected_profile else 2),
        },
        "generation_tokens": 256,
        "guard_tokens": 384,
        "planner_tokens": 768,
        "planner_request_bytes": 8192,
        "source_candidates": 80,
        "evidence_sources": 12,
        "source_body_bytes": 5000,
        "query_deadline_seconds": 3,
        "input_tokens": "conservative UTF-8 byte admission; not exact provider tokenization",
        "state": "one actual state-model update per profile; then 0/1/5 user turns lag",
        "memory_extractor": "not run: source-level working-memory experiment, no second engine",
        "grounding_prompt": GROUND_PROMPT,
        "answer_prompt": ANSWER_PROMPT,
        "stopping": "first technical failure/STOP/budget; no retry/resume/top-up",
        "database": "postgresql+asyncpg://hmph:hmph_local_only@127.0.0.1:15439/hmph",
        "product_default": "planner disabled",
    }
    if not execute:
        directory.mkdir(parents=True, exist_ok=False)
        for file in files:
            target = directory / "snapshot" / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
        save(directory / "plan.json", plan)
        print("Prepared three new product cases; real calls=0")
        return
    if (directory / "report.json").exists() or plan != json.loads(
        (directory / "plan.json").read_text(encoding="utf-8")
    ):
        raise ValueError("Frozen recall run changed or already started")
    budget = RunBudget(directory / "calls.sqlite", **plan["limits"])
    key = local_test_key(ROOT)
    report = {"status": "running", "rows": [], "human_review": "not_run"}

    def checkpoint():
        save(directory / "report.json", {**report, "budget": budget.snapshot()})

    checkpoint()
    try:
        for index, case in enumerate(cases, 1):
            if index not in plan["selected_cases"]:
                continue
            for profile in plan["profiles"]:
                services = Services(
                    key, budget, f"case-{index:03}", profile, control=RunControl(directory / "STOP")
                )
                schema = "recall_dev_" + uuid4().hex
                admin = create_async_engine(plan["database"])
                db = Database(plan["database"], schema=schema)
                row = {
                    "case": index,
                    "profile": profile,
                    "lag_user_turns": case["lag"],
                    "status": "running",
                    "stage": "setup",
                }
                report["rows"].append(row)
                try:
                    async with admin.begin() as connection:
                        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                    await db.migrate()
                    conv = CapturingConversation(
                        db,
                        digest_key=b"new-synthetic-recall-only-digest",
                        source_timezone="Asia/Hong_Kong",
                        recall_planner=services.model("planner").plan_recall
                        if profile == "focused"
                        else None,
                    )
                    identity = await conv.bootstrap()
                    history_chat = await conv.create_chat(
                        identity.owner_id, identity.character_id, "history"
                    )
                    current = await conv.create_chat(
                        identity.owner_id, identity.character_id, "current"
                    )
                    sources, message_ids = [], {}

                    async def add(
                        chat_id,
                        content,
                        mode="deepseek",
                        *,
                        conv=conv,
                        identity=identity,
                        sources=sources,
                        message_ids=message_ids,
                    ):
                        turn = await conv.accept(
                            identity.owner_id, chat_id, uuid4().hex, content, model_mode=mode
                        )
                        lease = await conv.claim("recall-seed", model_mode=mode)
                        if lease is None:
                            raise RuntimeError("Seed not claimed")
                        await conv.finish(lease)
                        source = next(
                            m
                            for m in await conv.messages(identity.owner_id, chat_id)
                            if m.turn_id == turn.id and m.role == "user"
                        )
                        sid = f"s{len(sources) + 1:03}"
                        message_ids[sid] = source.id
                        async with conv.db.sessions() as session:
                            observed = await session.scalar(
                                select(Turn.created_at).where(Turn.id == turn.id)
                            )
                        sources.append(Source(source_id=sid, observed_at=observed, content=content))

                    for content in case["history"]:
                        await add(history_chat.id, content, "fake")
                    await add(current.id, case["seed"])
                    dialogue = Dialogue(db)
                    row["stage"] = "state"
                    state_worker = StateWorker(
                        dialogue, services.model("state").update_state, model_mode="deepseek"
                    )
                    if not await state_worker.run_once():
                        raise RuntimeError("State task not claimed")
                    state = await dialogue.read(identity.owner_id, current.id)
                    row["state_before_lag"] = state.model_dump(mode="json")
                    if state.task_status != "idle" or state.error_code:
                        raise RuntimeError("State update failed")
                    for content in case["fillers"]:
                        await add(current.id, content)
                    services.check_stop()
                    query = await conv.accept(
                        identity.owner_id, current.id, "query", case["query"], model_mode="deepseek"
                    )
                    query_source = next(
                        m
                        for m in await conv.messages(identity.owner_id, current.id)
                        if m.turn_id == query.id and m.role == "user"
                    )
                    sid = f"s{len(sources) + 1:03}"
                    message_ids[sid] = query_source.id
                    worker = Worker(
                        conv,
                        services.model("generation"),
                        model_mode="deepseek",
                        model_name="deepseek-flash",
                        reviewer=services.model("guard").review,
                    )
                    began = time.monotonic()
                    row["stage"] = "generation"
                    await worker.run_once()
                    row["query_seconds"] = round(time.monotonic() - began, 4)
                    turn = await conv.get_turn(identity.owner_id, query.id)
                    row["turn"] = turn.model_dump(mode="json")
                    if turn.status != "completed" or not hasattr(conv, "actual_input"):
                        raise RuntimeError("Product reply not completed")
                    observed = conv.actual_input.reference_time
                    sources.append(
                        Source(source_id=sid, observed_at=observed, content=case["query"])
                    )
                    messages = await conv.messages(identity.owner_id, current.id)
                    reply = next(
                        m.content
                        for m in messages
                        if m.turn_id == query.id and m.role == "assistant"
                    )
                    scenario = SimpleNamespace(
                        history=tuple(sources),
                        query=Query(question=case["query"], timestamp=observed),
                    )
                    context = actual_context(conv.actual_input, scenario, message_ids, {})
                    row.update(
                        reply=reply,
                        actual_input=context,
                        memory_context=await conv.memory_explanation(identity.owner_id, query.id),
                    )
                    checkpoint()
                    row["stage"] = "grounding"
                    raw, _ = await services.model("grounding")._evaluation_json(
                        GROUND_PROMPT,
                        RootModel({"actual_input": context, "reply": reply}),
                        max_tokens=256,
                    )
                    row["grounding_raw"] = raw
                    grounding = GroundingScore.model_validate_json(raw)
                    row["stage"] = "answer_judge"
                    raw, _ = await services.model("answer_judge")._evaluation_json(
                        ANSWER_PROMPT,
                        RootModel(
                            {
                                "question": case["query"],
                                "reply": reply,
                                "reference": case["answer"],
                                "aliases": case["aliases"],
                            }
                        ),
                        max_tokens=256,
                    )
                    row["answer_raw"] = raw
                    answer = AnswerScore.model_validate_json(raw)
                    grade_case = SimpleNamespace(
                        gold=SimpleNamespace(
                            answer=case["answer"], evidence_sources=case["required"]
                        ),
                        evidence_alternatives=(),
                    )
                    row["grade"] = grade_product(grade_case, context, grounding, answer)
                    row["status"] = "completed"
                    services.check_budget()
                finally:
                    await services.close()
                    await db.close()
                    async with admin.begin() as connection:
                        await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
                    await admin.dispose()
                    row["schema_removed"] = True
                    checkpoint()
        report["status"] = "completed"
    except Exception as error:
        report.update(
            status="stopped" if (directory / "STOP").exists() else "failed",
            error_type=type(error).__name__,
            error_code=evaluation_error_code(error),
            failure_stage=row.get("stage") if report["rows"] else "initialization",
        )
    finally:
        checkpoint()
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": len(report["rows"]),
                "calls": len(budget.snapshot()["calls"]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--shared-threads", type=Path, help="isolated C2 demo directory")
    modes.add_argument(
        "--recall", type=Path, help="new isolated P4-c product development directory"
    )
    parser.add_argument("--recall-case", type=int, choices=(1, 2, 3))
    parser.add_argument("--recall-profile", choices=("lexical", "focused"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.recall and (args.recall_case or args.recall_profile):
        parser.error("Recall selection requires --recall")
    asyncio.run(
        recall_demo(
            args.recall,
            args.run,
            selected_case=args.recall_case,
            selected_profile=args.recall_profile,
        )
        if args.recall
        else shared_threads_demo(args.shared_threads, args.run)
    )
