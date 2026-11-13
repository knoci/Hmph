"""One fixed, real 30-turn product conversation. No human scoring material."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_run import CapturingConversation, Services, maintenance, save  # noqa: E402
from pydantic import SecretStr  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from hmph.config import Settings  # noqa: E402
from hmph.database import Database  # noqa: E402
from hmph.deepseek import DeepSeekModel  # noqa: E402
from hmph.eval_budget import RunBudget  # noqa: E402
from hmph.framework_worker import FrameworkWorker  # noqa: E402
from hmph.mem0_backend import Mem0Backend, Mem0SourceReader  # noqa: E402
from hmph.memory import Memory  # noqa: E402
from hmph.memory_builds import MemoryBuilds  # noqa: E402
from hmph.models import MemoryBuild  # noqa: E402
from hmph.online_memory import MemorySelection  # noqa: E402
from hmph.product_eval import RunControl  # noqa: E402
from hmph.runtime import build_worker  # noqa: E402
from hmph.shared_threads import SharedThreads  # noqa: E402
from hmph.timem_reference import (  # noqa: E402
    LocalTiMem,
    MeteredTiMemLLM,
    TiMemSupplement,
    build_snapshot_layers,
    install_provider,
    local_test_key,
)

TOPIC = "每周一起聊聊练琴进展"
POEM = "山居秋暝"
LIMITS = {"max_calls": 120, "max_input_bytes": 2 * 1024 * 1024, "per_case_calls": 120}
CANDIDATE_LIMITS = {"max_calls": 224, "max_input_bytes": 8 * 1024 * 1024, "per_case_calls": 128}
FIRST_CANDIDATE_ROUND = 25
SCENES = (
    {"user": "以后请叫我小沐。"},
    {"user": "我现在在青禾工作室做插画师。"},
    {"user": "我现在在哪儿工作？", "required": ("青禾",)},
    {"user": "更正一下，我已经离开青禾工作室，现在在远山设计做插画师。"},
    {"user": "那我现在工作的地方叫什么？只说现在的。", "required": ("远山",)},
    {"user": f"记下共同计划：{TOPIC}", "thread_status": "open"},
    {"user": "这个计划先从每天练十分钟开始。"},
    {"user": "咱们共同跟进的是哪件事？", "required": ("琴",)},
    {"user": "今天练了十分钟，左手还是有点僵。"},
    {"user": f"暂停共同计划：{TOPIC}", "thread_status": "paused"},
    {"user": "先不聊练琴了，讲个关于猫的小笑话。"},
    {"user": "帮我写一句关于春天的短句。"},
    {"user": "我回来了，先按约定叫我。", "new_chat": True, "required": ("小沐",)},
    {"user": "以后请叫我小岚。"},
    {"user": "按新称呼叫我，再说句简短的打气话。", "required": ("小岚",)},
    {"user": f"恢复共同计划：{TOPIC}", "thread_status": "open"},
    {"user": "接着聊我们刚恢复的计划吧，这次想把左右手配合练顺。"},
    {"user": "顺便问一句，我现在工作的地方叫什么？", "required": ("远山",)},
    {"user": "今天有点累，只想随便聊两句。"},
    {"user": "你这嘴硬劲儿，还挺像个有脾气的小家伙。"},
    {"user": f"我最近在背一首诗：《{POEM}》。", "remember_source": True},
    {"user": "刚才说的那首诗叫什么？", "required": (POEM,)},
    {"user": "今天不背诗了，给我一个关于月亮的比喻。"},
    {"user": "讲个两三句的小故事，主角是一只不想起床的狐狸。"},
    {
        "user": "又开个新会话。按最新约定叫我，再说说我现在在哪儿工作。",
        "new_chat": True,
        "required": ("小岚", "远山"),
    },
    {"user": "我们现在还有什么共同计划？", "required": ("琴",)},
    {
        "user": "我刚在记忆管理里删掉了那首诗的来源，现在还记得它的题目吗？",
        "forget_source": True,
    },
    {"user": "那就不回头找了。换个话题，给新来的小猫取两个名字。"},
    {"user": "先按最新称呼叫我，再确认我现在在哪工作。", "required": ("小岚", "远山")},
    {
        "user": "新会话也别忘了：叫我什么、现在在哪工作、我们还在跟进什么计划？",
        "new_chat": True,
        "required": ("小岚", "远山", "琴"),
    },
)


async def run(directory, *, use_candidate=False):
    directory.mkdir(parents=True, exist_ok=False)
    profile = "mem0_timem" if use_candidate else "lexical"
    limits = CANDIDATE_LIMITS if use_candidate else LIMITS
    control = RunControl(directory / "STOP")
    plan = {
        "purpose": "user-requested 30-turn product self-check before any human review",
        "model_mode": "deepseek",
        "model": "deepseek-flash",
        "temperature": 0,
        "backend": profile,
        "limits": limits,
        "native_writer_limits": {"mem0": 32, "timem": 60} if use_candidate else None,
        "candidate_starts_at": FIRST_CANDIDATE_ROUND if use_candidate else None,
        "generation_network_retries": 1,
        "other_retries": 0,
        "generation_tokens": 256,
        "turn_timeout_seconds": 45,
        "scenes": SCENES,
        "acceptance": "30 published real replies; required facts, sources and controls pass",
        "code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), *sorted((ROOT / "src/hmph").glob("*.py")))
        },
    }
    save(directory / "plan.json", plan)
    budget = RunBudget(directory / "budget.sqlite3", **limits)
    report = {
        "status": "running",
        "model_mode": "deepseek",
        "rows": [],
        "background_failures": [],
        "schema_removed": False,
        "candidate": None,
        "human_review": "stopped",
    }

    def checkpoint():
        report["budget"] = budget.snapshot()
        save(directory / "report.json", report)

    url = os.getenv(
        "HMPH_SELF_CHECK_DATABASE_URL",
        "postgresql+asyncpg://hmph:hmph_local_only@127.0.0.1:15440/hmph",
    )
    schema = "product_check_" + uuid4().hex
    db, admin = Database(url, schema=schema), create_async_engine(url)
    key = local_test_key(ROOT)
    settings = Settings(
        _env_file=None,
        api_token=SecretStr("isolated-self-check-api-token-only"),
        request_digest_key=SecretStr("isolated-self-check-digest-key-only"),
        model_mode="deepseek",
        deepseek_api_key=key,
        deepseek_model=plan["model"],
        lease_seconds=30,
    )
    services = Services(key, budget, "case-001", profile, control=control)
    builds = framework = timem = llm = build_id = None
    candidate_cleaned = False

    async def clean_candidate(owner):
        nonlocal candidate_cleaned
        if build_id is None or candidate_cleaned:
            return
        if getattr(timem, "client", None):
            report["candidate"]["timem_cleaned"] = timem.close()
            timem.client = None
        async with db.sessions.begin() as session:
            await builds._retire(session, [build_id])
        await framework.run_once(schedule_maintenance=False)
        spaces = await builds.spaces(owner, build_id)
        report["candidate"]["mem0_cleaned"] = all(space.status == "cleaned" for space in spaces)
        candidate_cleaned = report["candidate"]["mem0_cleaned"] and report["candidate"].get(
            "timem_cleaned", True
        )

    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        await db.migrate()
        conversation = CapturingConversation(
            db,
            digest_key=settings.request_digest_key.get_secret_value().encode(),
            source_timezone="Asia/Hong_Kong",
        )
        identity = await conversation.bootstrap()
        if use_candidate:
            builds = MemoryBuilds(db, digest_key=conversation.digest_key)
            framework = FrameworkWorker(
                builds,
                Mem0Backend(
                    key,
                    database_url=url,
                    max_calls=32,
                    budget=budget.bind("case-002", "mem0", "writer"),
                    stop_file=control.stop_file,
                ),
            )
            timem = LocalTiMem()
            llm = MeteredTiMemLLM(
                key, budget, "case-003", max_calls=60, stop_file=control.stop_file
            )
        chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
        worker = build_worker(conversation, settings)
        assert worker.model_mode == "deepseek" and isinstance(worker.model, DeepSeekModel)
        assert worker.reviewer is None
        # Same real adapter and model, with the existing durable metering transport.
        worker.model = services.model("generation")
        services.model("state")
        state_transport = services.transports["state"]
        state_transport.capture_response = True

        def capture_state_failure(lease, error):
            filename = f"state-failure-{len(report['background_failures'])}.json"
            save(
                directory / filename,
                {
                    "failure": error,
                    "lease": lease.model_dump(mode="json"),
                    "response": state_transport.last_response,
                },
            )

        threads = SharedThreads(conversation)
        memory = Memory(db, digest_key=conversation.digest_key)
        poem_source = None
        for number, scene in enumerate(SCENES, 1):
            control.check()
            if use_candidate and number == FIRST_CANDIDATE_ROUND:
                requested = await builds.request(
                    identity.owner_id, identity.character_id, "self-check-snapshot", backend="mem0"
                )
                build_id = requested.id
                report["candidate"] = {"build_id": str(build_id), "status": "building_mem0"}
                checkpoint()
                await framework.run_once(schedule_maintenance=False)
                build = await builds.get(identity.owner_id, build_id)
                if build.status != "ready":
                    report["candidate"]["error"] = build.error_code or build.status
                    raise RuntimeError("self_check_mem0_build_failed")
                space = (await builds.spaces(identity.owner_id, build_id))[0]
                report["candidate"].update(status="building_timem", mem0_usage=space.usage)
                checkpoint()
                async with db.sessions() as session:
                    frozen = await builds._sources(
                        session, await session.get(MemoryBuild, build_id)
                    )
                install_provider(llm)
                await timem.open(directory)
                await build_snapshot_layers(timem, frozen)
                await MemorySelection(
                    db, digest_key=conversation.digest_key, allow_mem0=True
                ).update(
                    identity.owner_id,
                    identity.character_id,
                    "self-check-select",
                    expected_revision=1,
                    build_id=build_id,
                )
                mem0 = Mem0SourceReader(
                    url, model_cache=ROOT / "artifacts/models/multilingual-minilm"
                )
                conversation.source_reader = TiMemSupplement(
                    mem0, timem, build_id=build_id, space_id=space.id
                )
                report["candidate"].update(status="ready", sources=len(frozen))
                del frozen
            row = {"number": number, **scene, "status": "running"}
            report["rows"].append(row)
            if scene.get("new_chat"):
                chat = await conversation.create_chat(
                    identity.owner_id, identity.character_id, uuid4().hex
                )
            row["chat_id"] = str(chat.id)
            if scene.get("forget_source"):
                preview = await memory.preview_forget_source(identity.owner_id, poem_source.id)
                receipt = await memory.forget_source(
                    identity.owner_id,
                    poem_source.id,
                    uuid4().hex,
                    expected_revision=poem_source.revision,
                )
                if use_candidate:
                    await clean_candidate(identity.owner_id)
                    receipt = await memory.deletion(identity.owner_id, receipt.id)
                row["forget_preview"] = preview.model_dump(mode="json")
                row["forget_receipt"] = receipt.model_dump(mode="json")
            checkpoint()
            accepted = await conversation.accept(
                identity.owner_id, chat.id, uuid4().hex, scene["user"], model_mode="deepseek"
            )
            conversation.actual_input = None
            await worker.run_once()
            saved = await conversation.get_turn(identity.owner_id, accepted.id)
            messages = await conversation.messages(identity.owner_id, chat.id)
            row["turn"] = saved.model_dump(mode="json")
            row["reply"] = next(
                (m.content for m in messages if m.turn_id == accepted.id and m.role == "assistant"),
                "",
            )
            request = conversation.actual_input
            row["input"] = request.model_dump(mode="json") if request is not None else None
            workspace = request.workspace if request is not None else None
            row["read_backend"] = workspace.backend if workspace else None
            row["status"] = saved.status
            row["required_met"] = all(s in row["reply"] for s in scene.get("required", ()))
            row["real_mode"] = (
                saved.model_mode == "deepseek"
                and (saved.usage or {}).get("provider") == "deepseek"
                and (saved.usage or {}).get("guard_action") == "not_requested"
            )
            if scene.get("remember_source"):
                poem_source = next(
                    m for m in messages if m.turn_id == accepted.id and m.role == "user"
                )
            if scene.get("thread_status"):
                row["threads"] = await threads.read(identity.owner_id, identity.character_id)
                row["thread_state_met"] = any(
                    t["topic"] == TOPIC and t["status"] == scene["thread_status"]
                    for t in row["threads"]["items"]
                )
            if scene.get("forget_source"):
                row["forget_met"] = (
                    row["forget_receipt"]["status"] == "completed"
                    and POEM not in row["reply"]
                    and POEM not in json.dumps(row["input"], ensure_ascii=False)
                )
            checkpoint()
            print(
                json.dumps(
                    {"round": number, "user": scene["user"], "reply": row["reply"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if saved.status != "completed":
                raise RuntimeError("technical_chat_failure")
            await maintenance(
                db, services, report["background_failures"], state_failure=capture_state_failure
            )
            checkpoint()
        rows = report["rows"]
        report["acceptance"] = {
            "thirty_completed": len(rows) == 30 and all(r["status"] == "completed" for r in rows),
            "all_real_deepseek": all(r["real_mode"] for r in rows),
            "required_facts": all(r["required_met"] for r in rows),
            "plan_pause_resume": all(r.get("thread_state_met", True) for r in rows),
            "forget_source": all(r.get("forget_met", True) for r in rows),
            "background_success": not report["background_failures"],
        }
        if use_candidate:
            report["acceptance"]["candidate_read_used"] = any(
                r["read_backend"] == "mem0_timem" for r in rows[FIRST_CANDIDATE_ROUND - 1 : 26]
            )
            report["acceptance"]["candidate_forgotten"] = candidate_cleaned
        report["status"] = "passed" if all(report["acceptance"].values()) else "failed"
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error_code=getattr(
                error,
                "code",
                str(error)
                if isinstance(error, RuntimeError) and str(error).startswith("self_check_")
                else "local_self_check_failure",
            ),
        )
    finally:
        if build_id is not None:
            try:
                await clean_candidate(identity.owner_id)
            except Exception as error:
                report["candidate"]["cleanup_error"] = type(error).__name__
                report["status"] = "failed"
        if llm is not None:
            await llm.close()
        await services.close()
        await db.close()
        if build_id is None or candidate_cleaned:
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            report["schema_removed"] = True
        await admin.dispose()
        checkpoint()
    print(
        json.dumps(
            {"status": report["status"], "acceptance": report.get("acceptance")}, ensure_ascii=False
        )
    )
    return report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute the paid, fixed batch")
    parser.add_argument("--timem-mem0", action="store_true", help="enable the isolated candidate")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/live/product-self-check-30"
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required; this command makes paid local model calls")
    if args.timem_mem0:
        sys.path.insert(0, str(ROOT / "artifacts/references/TiMem"))
        sys.path.insert(0, str(ROOT / "artifacts/references/mem0"))
        sys.path.append(str(ROOT / "artifacts/framework-env/Lib/site-packages"))
    raise SystemExit(0 if asyncio.run(run(args.output, use_candidate=args.timem_mem0)) else 1)
