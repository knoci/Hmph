"""Real DeepSeek end-to-end verification for the standalone SQLite local memory backend.

Runs three isolated scenarios with real close/restart cycles:
  A. Standard piano lifecycle + independent running plan + 2 restarts + forget
  B. Colloquial paraphrase lifecycle ("法语考试先搁一搁吧，这阵子别继续追问了") + independent plant plan
  C. Ambiguous reference protection across two similar exam plans

Saves full transcript, replies, version chains, and exact user quotes to a JSON report.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from hmph.local_memory import LocalMemoryEngine, OpenAICompatLLM, resolve_llm_config


def serialize_items(engine: LocalMemoryEngine) -> list[dict]:
    return [item.model_dump(mode="json") for item in engine.list_items()]


def make_engine(db_path: Path, llm: OpenAICompatLLM, embedder, tz: str) -> LocalMemoryEngine:
    return LocalMemoryEngine(db_path, llm=llm, embedder=embedder, timezone_name=tz)


async def run_scenario_a(db_path: Path, llm: OpenAICompatLLM, embedder, tz: str) -> dict:
    if db_path.exists():
        db_path.unlink()
    transcript: list[dict] = []

    engine = make_engine(db_path, llm, embedder, tz)

    async def say(eng: LocalMemoryEngine, sid: str, text: str, label: str) -> dict:
        turn = await eng.chat(sid, text, await_extraction=True)
        entry = {
            "step": label,
            "session": sid,
            "user": text,
            "reply": turn.reply,
            "intent": turn.evidence.query_intent,
            "retrieval_ms": turn.evidence.retrieval_ms,
            "generation_ms": turn.generation_ms,
            "used_item_ids": list(turn.evidence.used_item_ids),
            "narrated_context": turn.evidence.narrated_context,
            "usage": turn.usage,
        }
        transcript.append(entry)
        print(f"  [{label}] OK ({turn.generation_ms}ms)", flush=True)
        return entry

    await say(engine, "s1", "记一下，我们约好每周六一起跟进练琴进展。", "A1_create_piano")
    await say(engine, "s1", "另外再记一个单独的计划，每周三晚上我要坚持夜跑五公里。", "A2_create_running")
    await say(engine, "s1", "练琴这周起改到周日吧，周六我经常起不来。", "A3_reschedule_piano")
    await say(engine, "s1", "算了，练琴先搁一搁吧，这阵子别继续催我了。", "A4_pause_piano")
    casual = await say(engine, "s2", "今天中午好纠结啊，吃拉面还是吃咖喱饭好？", "A5_casual_new_topic")

    items_before_restart = serialize_items(engine)
    await engine.flush()
    engine.store.close()

    # Restart #1: brand-new LocalMemoryEngine & SQLite connection on the exact same .db file.
    engine2 = make_engine(db_path, llm, embedder, tz)
    ask_both = await say(engine2, "s3", "练琴原来约的是哪天？现在还练吗？", "A6_after_restart_ask_history_and_current")
    await say(engine2, "s3", "最近没那么忙了，练琴恢复吧，继续按之前改后的时间来。", "A7_resume_piano")
    overview = await say(engine2, "s3", "现在我还在进行的计划有哪些，分别怎么安排的？", "A8_check_active_plans")

    items_after_resume = serialize_items(engine2)
    forget_receipt = await engine2.forget("练琴")
    await engine2.flush()
    engine2.store.close()

    # Restart #2 after forgetting: brand-new engine & SQLite connection.
    engine3 = make_engine(db_path, llm, embedder, tz)
    after_forget = await say(engine3, "s4", "我们之前约过练琴吗？原来约哪天、现在还练吗？", "A9_after_forget_and_restart")
    other_intact = await say(engine3, "s4", "那我还有什么正在进行的计划吗？", "A10_verify_other_plan_intact")
    items_after_forget = serialize_items(engine3)
    await engine3.flush()
    engine3.store.close()

    return {
        "name": "Scenario A: 标准练琴闭环 + 独立夜跑计划 + 两次重启 + 遗忘",
        "db_path": str(db_path),
        "embedding_backend": embedder.name,
        "transcript": transcript,
        "items_before_restart": items_before_restart,
        "items_after_resume": items_after_resume,
        "forget_receipt": forget_receipt.model_dump(mode="json"),
        "items_after_forget": items_after_forget,
        "checks": {
            "casual_no_piano_nag": "练琴" not in casual["reply"] and "练琴" not in casual["narrated_context"],
            "history_mentions_sat_and_sun": ("周六" in ask_both["reply"]) and ("周日" in ask_both["reply"]),
            "current_mentions_paused": any(w in ask_both["reply"] for w in ("暂停", "搁置", "搁一搁", "没在练", "停")),
            "overview_has_both_plans": ("练琴" in overview["reply"]) and ("夜跑" in overview["reply"] or "跑步" in overview["reply"]),
            "after_forget_no_piano_memory": ("练琴" not in after_forget["narrated_context"]) and ("周六" not in after_forget["reply"]) and ("周日" not in after_forget["reply"]),
            "other_plan_still_active": any(("夜跑" in it["title"] or "跑步" in it["title"]) and it["status"] == "active" for it in items_after_forget),
            "other_plan_answered": ("夜跑" in other_intact["reply"] or "跑步" in other_intact["reply"]),
        },
    }


async def run_scenario_b(db_path: Path, llm: OpenAICompatLLM, embedder, tz: str) -> dict:
    if db_path.exists():
        db_path.unlink()
    transcript: list[dict] = []

    engine = make_engine(db_path, llm, embedder, tz)

    async def say(eng: LocalMemoryEngine, sid: str, text: str, label: str) -> dict:
        turn = await eng.chat(sid, text, await_extraction=True)
        entry = {
            "step": label,
            "session": sid,
            "user": text,
            "reply": turn.reply,
            "intent": turn.evidence.query_intent,
            "retrieval_ms": turn.evidence.retrieval_ms,
            "generation_ms": turn.generation_ms,
            "narrated_context": turn.evidence.narrated_context,
            "usage": turn.usage,
        }
        transcript.append(entry)
        print(f"  [{label}] OK ({turn.generation_ms}ms)", flush=True)
        return entry

    await say(engine, "s1", "帮我记个事，我打算每周二晚上复习法语考试。", "B1_create_french")
    await say(engine, "s1", "顺便记一下，每个月一号我都要给家里的绿植换一次营养液。", "B2_create_plants")
    await say(engine, "s1", "周二晚上老加班，法语复习挪到周四晚上吧。", "B3_reschedule_french")
    await say(engine, "s1", "法语考试先搁一搁吧，这阵子别继续追问了。", "B4_colloquial_pause_french")
    casual = await say(engine, "s2", "刚才外面突然下大雨了，没带伞好烦啊。", "B5_casual_new_topic")

    items_paused = serialize_items(engine)
    await engine.flush()
    engine.store.close()

    # Restart #1
    engine2 = make_engine(db_path, llm, embedder, tz)
    ask_both = await say(engine2, "s3", "法语复习最初定的是礼拜几？眼下还在继续吗？", "B6_after_restart_ask")
    await say(engine2, "s3", "手头项目收尾了，法语考试重新捡起来吧。", "B7_colloquial_resume")
    items_resumed = serialize_items(engine2)
    forget_receipt = await engine2.forget("法语")
    await engine2.flush()
    engine2.store.close()

    # Restart #2
    engine3 = make_engine(db_path, llm, embedder, tz)
    after_forget = await say(engine3, "s4", "法语考试之前定在周几复习来着？现在呢？", "B8_after_forget_restart")
    other_ok = await say(engine3, "s4", "给绿植换营养液那个安排还在吗？", "B9_other_plan_intact")
    items_after_forget = serialize_items(engine3)
    await engine3.flush()
    engine3.store.close()

    return {
        "name": "Scenario B: 口语改述闭环（含“法语考试先搁一搁吧，这阵子别继续追问了”）",
        "db_path": str(db_path),
        "embedding_backend": embedder.name,
        "transcript": transcript,
        "items_paused": items_paused,
        "items_resumed": items_resumed,
        "forget_receipt": forget_receipt.model_dump(mode="json"),
        "items_after_forget": items_after_forget,
        "checks": {
            "french_paused_in_db": any("法语" in it["title"] and it["status"] == "paused" for it in items_paused),
            "casual_no_french_nag": "法语" not in casual["reply"] and "法语" not in casual["narrated_context"],
            "history_mentions_tue_and_thu": ("周二" in ask_both["reply"] or "礼拜二" in ask_both["reply"]) and ("周四" in ask_both["reply"] or "礼拜四" in ask_both["reply"]),
            "current_mentions_paused": any(w in ask_both["reply"] for w in ("暂停", "搁置", "搁一搁", "停", "没在")),
            "resumed_in_db": any("法语" in it["title"] and it["status"] == "active" for it in items_resumed),
            "after_forget_clean": (
                ("法语" not in after_forget["narrated_context"])
                and not any(
                    d in after_forget["reply"]
                    for d in ("周一", "周二", "周三", "周四", "周五", "周六", "周日", "礼拜")
                )
                and any(w in after_forget["reply"] for w in ("没印象", "不记得", "没约", "没有", "记不得", "没听"))
            ),
            "plants_still_active": any("绿植" in it["title"] or "营养液" in it["title"] for it in items_after_forget) and ("一号" in other_ok["reply"] or "1号" in other_ok["reply"]),
        },
    }


async def run_scenario_c(db_path: Path, llm: OpenAICompatLLM, embedder, tz: str) -> dict:
    if db_path.exists():
        db_path.unlink()
    transcript: list[dict] = []

    engine = make_engine(db_path, llm, embedder, tz)

    async def say(eng: LocalMemoryEngine, sid: str, text: str, label: str) -> dict:
        turn = await eng.chat(sid, text, await_extraction=True)
        entry = {
            "step": label,
            "session": sid,
            "user": text,
            "reply": turn.reply,
            "narrated_context": turn.evidence.narrated_context,
            "usage": turn.usage,
        }
        transcript.append(entry)
        print(f"  [{label}] OK ({turn.generation_ms}ms)", flush=True)
        return entry

    await say(engine, "s1", "这阵子我要准备法语考试，帮我记着。", "C1_french_exam")
    await say(engine, "s1", "同时我也在准备驾照考试，也帮我记一下。", "C2_driving_exam")
    amb_turn = await say(engine, "s1", "考试先放一放吧，最近实在顾不过来。", "C3_ambiguous_pause")

    items = serialize_items(engine)
    ambiguities = [a.model_dump(mode="json") for a in engine.list_open_ambiguities()]
    await engine.flush()
    engine.store.close()

    active_exams = [it for it in items if "考试" in it["title"] and it["status"] == "active"]
    return {
        "name": "Scenario C: 同类/模糊指代防误改保护",
        "db_path": str(db_path),
        "embedding_backend": embedder.name,
        "transcript": transcript,
        "items": items,
        "ambiguities": ambiguities,
        "checks": {
            "both_exams_untouched_active": len(active_exams) == 2,
            "ambiguity_recorded": len(ambiguities) >= 1,
            "reply_asks_clarification": (
                ("法语" in amb_turn["reply"] and "驾照" in amb_turn["reply"])
                or any(w in amb_turn["reply"] for w in ("哪一", "哪个", "哪场", "哪门", "说的哪"))
            ),
        },
    }


async def main() -> int:
    import argparse

    from hmph.local_memory import build_default_embedder

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=["a", "b", "c"], default=["a", "b", "c"])
    args = parser.parse_args()

    out_dir = Path("artifacts/local-memory/verify-20261121")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"

    api_key, base_url, model, tz = resolve_llm_config()
    llm = OpenAICompatLLM(api_key, base_url=base_url, model=model, temperature=0.2)
    embedder = build_default_embedder()
    print(f"Running scenarios={args.only} with model={model}, embedder={embedder.name}", flush=True)

    report: dict = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists() and set(args.only) != {"a", "b", "c"}
        else {}
    )
    try:
        if "a" in args.only:
            report["scenario_a"] = await run_scenario_a(out_dir / "scenario_a.db", llm, embedder, tz)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if "b" in args.only:
            report["scenario_b"] = await run_scenario_b(out_dir / "scenario_b.db", llm, embedder, tz)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if "c" in args.only:
            report["scenario_c"] = await run_scenario_c(out_dir / "scenario_c.db", llm, embedder, tz)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        await llm.aclose()

    all_passed = True
    for key in ("scenario_a", "scenario_b", "scenario_c"):
        if key not in report:
            continue
        sc = report[key]
        print(f"\n==================== {sc['name']} ====================")
        print(f"Embedding backend: {sc['embedding_backend']}")
        for turn in sc["transcript"]:
            print(f"  [{turn['step']}] ({turn['session']}) 用户: {turn['user']}")
            print(f"         Hmph: {turn['reply']}")
        print("  --- 检查结果 ---")
        for ck, ok in sc["checks"].items():
            mark = "PASS" if ok else "FAIL"
            if not ok:
                all_passed = False
            print(f"    [{mark}] {ck}")

    print(f"\n完整报告已保存至: {report_path.resolve()}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
