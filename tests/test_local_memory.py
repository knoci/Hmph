"""Offline unit tests for the standalone single-file SQLite local memory backend."""

from __future__ import annotations

import asyncio

import pytest

from hmph.local_memory import (
    ExtractedChange,
    ExtractionResponse,
    HashingEmbedder,
    LocalMemoryEngine,
)


class ScriptedLLM:
    """Deterministic LLM double for testing full memory lifecycle without API calls."""

    model_name = "scripted-mock"

    def __init__(self, *, extract_delay: float = 0.0):
        self.extract_delay = extract_delay
        self.seen_contexts: list[str] = []

    async def extract_changes(
        self,
        *,
        user_message,
        recent_messages,
        candidate_items,
        now_iso,
        timezone_name,
    ):
        if self.extract_delay:
            await asyncio.sleep(self.extract_delay)
        piano = next((it for it in candidate_items if "练琴" in it.title), None)
        run = next((it for it in candidate_items if "跑步" in it.title), None)
        exams = [it for it in candidate_items if "考试" in it.title]

        if "周六" in user_message and "练琴" in user_message:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="create",
                            title="练琴计划",
                            kind="plan",
                            summary="约好每周六练琴",
                            effective_time_text="周六",
                            entities=["练琴", "周六"],
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 50},
            )
        if "周三" in user_message and "跑步" in user_message:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="create",
                            title="周三夜跑",
                            kind="plan",
                            summary="每周三晚上跑步五公里",
                            effective_time_text="周三晚上",
                            entities=["跑步", "周三"],
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 50},
            )
        if "法语考试" in user_message and "准备" in user_message:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="create",
                            title="法语考试",
                            kind="plan",
                            summary="准备法语考试",
                            entities=["法语考试", "考试"],
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 50},
            )
        if "驾照考试" in user_message and "准备" in user_message:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="create",
                            title="驾照考试",
                            kind="plan",
                            summary="准备驾照考试",
                            entities=["驾照考试", "考试"],
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 50},
            )
        if "改到周日" in user_message and piano:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="reschedule",
                            target_item_id=piano.id,
                            title=piano.title,
                            summary="改到周日练琴",
                            effective_time_text="周日",
                            entities=["练琴", "周日"],
                            source_quote="练琴改到周日吧",
                        )
                    ]
                ),
                {"total_tokens": 50},
            )
        if "搁一搁" in user_message and "练琴" in user_message and piano:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="pause",
                            target_item_id=piano.id,
                            title=piano.title,
                            summary="练琴计划先搁一搁，暂不进行",
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 50},
            )
        if "恢复" in user_message and "练琴" in user_message and piano:
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="resume",
                            target_item_id=piano.id,
                            title=piano.title,
                            summary="恢复周日练琴",
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 50},
            )
        if user_message.strip() == "考试先放一放吧" and len(exams) >= 2:
            # Simulate ambiguous reference across two exam items.
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="ambiguous",
                            candidate_item_ids=[it.id for it in exams],
                            ambiguity_note="同时存在法语考试和驾照考试，未指明暂停哪一门",
                            source_quote=user_message,
                        )
                    ]
                ),
                {"total_tokens": 50},
            )
        if user_message.strip() == "考试都先停" and len(exams) >= 2:
            # Even if LLM mistakenly tries to mutate two items without plural cue, guard catches it.
            return (
                ExtractionResponse(
                    changes=[
                        ExtractedChange(
                            action="pause",
                            target_item_id=it.id,
                            summary=f"暂停{it.title}",
                            source_quote=user_message,
                        )
                        for it in exams
                    ]
                ),
                {"total_tokens": 50},
            )
        _ = run
        return ExtractionResponse(changes=[]), {"total_tokens": 20}

    async def generate_reply(
        self,
        *,
        user_message,
        recent_messages,
        narrated_context,
        now_iso,
        timezone_name,
    ):
        self.seen_contexts.append(narrated_context)
        return f"[mock reply to: {user_message}]", {"total_tokens": 30}


@pytest.mark.asyncio
async def test_full_lifecycle_restart_isolation_and_forget(tmp_path):
    db_path = tmp_path / "hmph.db"
    llm = ScriptedLLM()
    engine = LocalMemoryEngine(db_path, llm=llm, embedder=HashingEmbedder())

    # 1. Create two independent plans: piano + running.
    await engine.chat("s1", "我们约好每周六练琴吧", await_extraction=True)
    await engine.chat("s1", "另外每周三晚上跑步五公里也记一下", await_extraction=True)
    # 2. Reschedule piano Saturday -> Sunday.
    await engine.chat("s1", "这周起练琴改到周日吧", await_extraction=True)
    # 3. Pause piano orally.
    await engine.chat("s1", "练琴先搁一搁吧，这阵子别催我了", await_extraction=True)

    # Verify version chain & provenance.
    items = {it.title: it for it in engine.list_items()}
    assert items["练琴计划"].status == "paused"
    assert [v.change_type for v in items["练琴计划"].versions] == ["create", "reschedule", "pause"]
    assert items["练琴计划"].versions[0].effective_time_text == "周六"
    assert items["练琴计划"].versions[1].effective_time_text == "周日"
    assert "周六练琴" in items["练琴计划"].versions[0].source_quote
    assert items["周三夜跑"].status == "active"

    # 4. Change topic in a new session: paused plan must NOT be injected to nag the user.
    casual = await engine.chat("s2", "今天中午吃拉面还是盖浇饭好", await_extraction=True)
    assert "练琴" not in casual.evidence.narrated_context
    assert items["练琴计划"].id not in casual.evidence.used_item_ids

    # 5. Real close & restart on the same SQLite file.
    await engine.close()
    llm2 = ScriptedLLM()
    engine2 = LocalMemoryEngine(db_path, llm=llm2, embedder=HashingEmbedder())

    # 6. Ask both history ("原来约哪天") and current ("现在还练吗").
    ask = await engine2.chat("s3", "练琴原来约哪天，现在还练吗？", await_extraction=True)
    assert ask.evidence.query_intent == "both"
    ctx = ask.evidence.narrated_context
    assert "周六" in ctx and "周日" in ctx and "已暂停" in ctx
    assert "v1 [create -> active]" in ctx and "v2 [reschedule -> active]" in ctx

    # 7. Resume piano plan -> inherits Sunday effective time.
    await engine2.chat("s3", "好了，练琴恢复吧", await_extraction=True)
    resumed = {it.title: it for it in engine2.list_items()}["练琴计划"]
    assert resumed.status == "active"
    assert resumed.current is not None and resumed.current.effective_time_text == "周日"
    assert len(resumed.versions) == 4

    # 8. Forget piano plan atomically, then restart again.
    receipt = await engine2.forget("练琴")
    assert len(receipt.deleted_item_ids) == 1
    assert len(receipt.deleted_version_ids) == 4
    assert len(receipt.deleted_message_ids) >= 4
    await engine2.close()

    llm3 = ScriptedLLM()
    engine3 = LocalMemoryEngine(db_path, llm=llm3, embedder=HashingEmbedder())
    remaining = {it.title: it for it in engine3.list_items()}
    assert "练琴计划" not in remaining
    assert "周三夜跑" in remaining and remaining["周三夜跑"].status == "active"

    after_forget = await engine3.chat("s4", "练琴原来约哪天，现在还练吗？", await_extraction=True)
    assert "练琴" not in after_forget.evidence.narrated_context
    assert "周六" not in after_forget.evidence.narrated_context
    assert "周日" not in after_forget.evidence.narrated_context
    await engine3.close()


@pytest.mark.asyncio
async def test_pending_raw_utterance_overrides_stale_state(tmp_path):
    """When background extraction has not finished yet, pending raw utterance takes priority."""
    db_path = tmp_path / "pending.db"
    llm = ScriptedLLM(extract_delay=0.25)
    engine = LocalMemoryEngine(db_path, llm=llm, embedder=HashingEmbedder())

    await engine.chat("s1", "我们约好每周六练琴吧", await_extraction=True)
    # Trigger a change WITHOUT waiting for background extraction.
    await engine.chat("s1", "这周起练琴改到周日吧", await_extraction=False)
    # Immediately ask in the next turn while extraction is still pending!
    turn = await engine.chat("s1", "练琴现在约在周几？", await_extraction=False)
    assert len(turn.evidence.pending_message_ids) >= 1
    assert "刚说的最新原话" in turn.evidence.narrated_context
    assert "练琴改到周日吧" in turn.evidence.narrated_context
    await engine.close()


@pytest.mark.asyncio
async def test_ambiguous_reference_does_not_corrupt_similar_items(tmp_path):
    """Ambiguous utterance touching two similar items is preserved as ambiguity, not applied."""
    db_path = tmp_path / "ambiguous.db"
    llm = ScriptedLLM()
    engine = LocalMemoryEngine(db_path, llm=llm, embedder=HashingEmbedder())

    await engine.chat("s1", "最近在准备法语考试", await_extraction=True)
    await engine.chat("s1", "另外也在准备驾照考试", await_extraction=True)

    await engine.chat("s1", "考试先放一放吧", await_extraction=True)
    items = {it.title: it for it in engine.list_items()}
    assert items["法语考试"].status == "active"
    assert items["驾照考试"].status == "active"
    ambs = engine.list_open_ambiguities()
    assert len(ambs) == 1
    assert "考试先放一放吧" in ambs[0].source_quote
    assert len(ambs[0].candidate_item_ids) == 2

    # Next turn about exams surfaces the open ambiguity for clarification.
    check = await engine.chat("s1", "我刚才说的考试怎么样了？", await_extraction=True)
    assert "未决指代疑点" in check.evidence.narrated_context
    await engine.close()


@pytest.mark.asyncio
async def test_local_onnx_semantic_retrieval_and_restart(tmp_path):
    """Verify real local multilingual-minilm ONNX embeddings persist in SQLite and recall cross-paraphrase."""
    from pathlib import Path

    model_dir = Path("artifacts/models/multilingual-minilm")
    if not model_dir.exists():
        pytest.skip("Local multilingual-minilm cache not present")

    from hmph.local_memory import LocalFastEmbedder

    db_path = tmp_path / "onnx.db"
    try:
        embedder = LocalFastEmbedder(cache_dir=model_dir)
    except Exception as exc:
        pytest.skip(f"Local ONNX runtime unavailable in this environment: {exc}")
    engine = LocalMemoryEngine(db_path, llm=ScriptedLLM(), embedder=embedder)

    await engine.chat("s1", "我们约好每周六练琴吧", await_extraction=True)
    await engine.chat("s1", "这周起练琴改到周日吧", await_extraction=True)
    await engine.close()

    # Reopen the same single SQLite file and query using a paraphrase without exact overlap ("钢琴课").
    engine2 = LocalMemoryEngine(db_path, llm=ScriptedLLM(), embedder=embedder)
    hits = engine2.store.search_items_semantic("周末弹钢琴的安排是什么时候")
    assert len(hits) >= 1
    recalled = engine2.store.get_item(hits[0])
    assert recalled is not None and recalled.title == "练琴计划"
    await engine2.close()
