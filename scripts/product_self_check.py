"""One fixed, real 30-turn product conversation. No human scoring material."""

import argparse
import asyncio
import hashlib
import json
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
from hmph.memory import Memory  # noqa: E402
from hmph.runtime import build_worker  # noqa: E402
from hmph.shared_threads import SharedThreads  # noqa: E402
from hmph.timem_reference import local_test_key  # noqa: E402

TOPIC = "每周一起聊聊练琴进展"
POEM = "山居秋暝"
BANNED = ("尊重边界", "不想聊也没关系", "暂时无法提供", "请稍后再试", "系统繁忙")
LIMITS = {"max_calls": 120, "max_input_bytes": 2 * 1024 * 1024, "per_case_calls": 120}
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


async def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    plan = {
        "purpose": "user-requested 30-turn product self-check before any human review",
        "model_mode": "deepseek",
        "model": "deepseek-flash",
        "temperature": 0,
        "backend": "lexical",
        "limits": LIMITS,
        "generation_network_retries": 1,
        "other_retries": 0,
        "generation_tokens": 256,
        "turn_timeout_seconds": 45,
        "scenes": SCENES,
        "banned_reply_fragments": BANNED,
        "acceptance": "30 published real replies, no banned text, required facts and controls pass",
        "code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), *sorted((ROOT / "src/hmph").glob("*.py")))
        },
    }
    save(directory / "plan.json", plan)
    budget = RunBudget(directory / "budget.sqlite3", **LIMITS)
    report = {
        "status": "running",
        "model_mode": "deepseek",
        "rows": [],
        "background_failures": [],
        "schema_removed": False,
        "human_review": "stopped",
    }

    def checkpoint():
        report["budget"] = budget.snapshot()
        save(directory / "report.json", report)

    url = "postgresql+asyncpg://hmph:hmph_local_only@127.0.0.1:15440/hmph"
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
    services = Services(key, budget, "case-001", "lexical")
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
        chat = await conversation.create_chat(identity.owner_id, identity.character_id, uuid4().hex)
        worker = build_worker(conversation, settings)
        assert worker.model_mode == "deepseek" and isinstance(worker.model, DeepSeekModel)
        assert worker.reviewer is None
        # Same real adapter and model, with the existing durable metering transport.
        worker.model = services.model("generation")
        threads = SharedThreads(conversation)
        memory = Memory(db, digest_key=conversation.digest_key)
        poem_source = None
        for number, scene in enumerate(SCENES, 1):
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
                row["forget_preview"] = preview.model_dump(mode="json")
                row["forget_receipt"] = receipt.model_dump(mode="json")
            checkpoint()
            accepted = await conversation.accept(
                identity.owner_id, chat.id, uuid4().hex, scene["user"], model_mode="deepseek"
            )
            await worker.run_once()
            saved = await conversation.get_turn(identity.owner_id, accepted.id)
            messages = await conversation.messages(identity.owner_id, chat.id)
            row["turn"] = saved.model_dump(mode="json")
            row["reply"] = next(
                (m.content for m in messages if m.turn_id == accepted.id and m.role == "assistant"),
                "",
            )
            row["input"] = conversation.actual_input.model_dump(mode="json")
            row["status"] = saved.status
            row["banned_hits"] = [s for s in BANNED if s in row["reply"]]
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
            await maintenance(db, services, report["background_failures"])
            checkpoint()
        rows = report["rows"]
        report["acceptance"] = {
            "thirty_completed": len(rows) == 30 and all(r["status"] == "completed" for r in rows),
            "no_banned_replies": all(not r["banned_hits"] for r in rows),
            "all_real_deepseek": all(r["real_mode"] for r in rows),
            "required_facts": all(r["required_met"] for r in rows),
            "plan_pause_resume": all(r.get("thread_state_met", True) for r in rows),
            "forget_source": all(r.get("forget_met", True) for r in rows),
            "background_success": not report["background_failures"],
        }
        report["status"] = "passed" if all(report["acceptance"].values()) else "failed"
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error_code=getattr(error, "code", "local_self_check_failure"),
        )
    finally:
        await services.close()
        await db.close()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()
        report["schema_removed"] = True
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
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/live/product-self-check-30"
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required; this command makes paid local model calls")
    raise SystemExit(0 if asyncio.run(run(args.output)) else 1)
