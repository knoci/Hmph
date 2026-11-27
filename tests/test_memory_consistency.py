"""Regressions for durable extraction order and the evidence actually sent to a model."""

import asyncio
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from hmph.local_memory.embeddings import HashingEmbedder
from hmph.local_memory.engine import LocalMemoryEngine
from hmph.local_memory.gateway import create_airi_gateway_app
from hmph.local_memory.store import LocalMemoryStore
from hmph.local_memory.types import ExtractedChange, ExtractionResponse


class FixtureModel:
    model_name = "fixture"

    def __init__(self):
        self.calls = []
        self.context = ""
        self.a_stream_started = asyncio.Event()
        self.release_a_stream = asyncio.Event()
        self.fail_on = None

    async def extract_changes(self, *, user_message, candidate_items, recent_messages, **_):
        self.calls.append(user_message)
        if user_message == self.fail_on:
            raise RuntimeError("fixture provider error")
        if user_message == "练琴约周六":
            change = ExtractedChange(
                action="create", title="练琴", summary="周六", effective_time_text="周六",
                source_quote=user_message,
            )
        else:
            [item] = candidate_items
            expected_day, new_day = {
                "练琴改到周日": ("周六", "周日"),
                "那个再改到周一": ("周日", "周一"),
                "它改到周二": ("周六", "周二"),
            }[user_message]
            assert item.current.effective_time_text == expected_day
            assert recent_messages and recent_messages[-1].role in {"user", "assistant"}
            change = ExtractedChange(
                action="reschedule", target_item_id=item.id, summary=new_day,
                effective_time_text=new_day, source_quote=user_message,
            )
        return ExtractionResponse(changes=[change]), {}

    async def extract_session_changes(self, **_):
        return ExtractionResponse(changes=[
            ExtractedChange(action="create", title="练琴", summary="周六", source_turn_index=0),
            ExtractedChange(action="create", title="跑步", summary="周日", source_turn_index=0),
        ]), {"total_tokens": 17}

    async def proxy_chat_completion_stream(self, *, messages, **_):
        if messages[-1]["content"] == "练琴改到周日":
            self.a_stream_started.set()
            await self.release_a_stream.wait()
        yield {"choices": [{"delta": {"content": "收到"}}]}

    async def answer_from_context(self, *, context_text, **_):
        self.context = context_text
        return "fixture", {}


def make_engine(model, db=":memory:"):
    return LocalMemoryEngine(
        db, llm=model, embedder=HashingEmbedder(),
        auto_consolidate_every=0, auto_summarize_turns=0,
        enable_ppr=False, enable_two_hop=False, enable_listwise_rerank=False,
    )


class MemoryConsistencyTests(unittest.IsolatedAsyncioTestCase):
    def temporary_database(self):
        return Path(self.enterContext(TemporaryDirectory(dir="."))) / "hmph.db"

    async def test_streams_finish_in_reverse_without_reordering_extraction(self):
        model = FixtureModel()
        engine = make_engine(model)
        self.addAsyncCleanup(engine.close)
        app = create_airi_gateway_app(engine, gateway_api_key="")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            async def chat(text):
                return await client.post(
                    "/v1/chat/completions",
                    headers={"X-Hmph-Session-Id": "s1"},
                    json={"stream": True, "messages": [{"role": "user", "content": text}]},
                )

            await engine.ingest_user_message("s1", "练琴约周六")
            slow = asyncio.create_task(chat("练琴改到周日"))
            try:
                await asyncio.wait_for(model.a_stream_started.wait(), timeout=5)
                reply_b = await asyncio.wait_for(chat("那个再改到周一"), timeout=5)
                assert "data: [DONE]" in reply_b.text
                assert not slow.done()
                await engine.flush()
                [item] = engine.list_items()
                assert item.current.effective_time_text == "周一"
                assert [v.effective_time_text for v in item.versions] == ["周六", "周日", "周一"]
                assert model.calls == ["练琴约周六", "练琴改到周日", "那个再改到周一"]
            finally:
                model.release_a_stream.set()
                await asyncio.wait_for(slow, timeout=5)
            await chat("那个再改到周一")
            await engine.flush()
            [item] = engine.list_items()
            assert len(item.versions) == 3
            assert item.current.effective_time_text == "周一"
            assert model.calls == ["练琴约周六", "练琴改到周日", "那个再改到周一"]


    async def test_recovers_pending_in_receive_order_and_resolves_pronoun(self):
        db = self.temporary_database()
        before = FixtureModel()
        before.fail_on = "练琴约周六"
        engine = make_engine(before, db)
        try:
            # Reverse the timestamps so only persisted arrival order can produce the right state.
            await engine.ingest_user_message(
                "s1", "练琴约周六", now_iso="2026-11-27T10:00:02Z", await_extraction=False
            )
            await engine.ingest_user_message(
                "s1", "它改到周二", now_iso="2026-11-27T10:00:01Z", await_extraction=False
            )
            await engine.flush()
            assert before.calls == ["练琴约周六"]
            assert len(engine.store.get_pending_messages()) == 2
        finally:
            await engine.close()

        # Recreate the pre-upgrade schema to exercise sequence backfill during recovery.
        with sqlite3.connect(db) as connection:
            connection.execute("DROP TRIGGER messages_assign_ingest_order")
            connection.execute("DROP TABLE message_ingest_order")
        connection.close()

        after = FixtureModel()
        engine = make_engine(after, db)
        try:
            assert await engine.recover_pending_extractions() == 2
            await engine.flush()
            assert after.calls == ["练琴约周六", "它改到周二"]
            [item] = engine.list_items()
            assert [v.effective_time_text for v in item.versions] == ["周六", "周二"]
            assert item.current.effective_time_text == "周二"
            assert await engine.recover_pending_extractions() == 0
        finally:
            await engine.close()


    async def test_batch_extraction_commits_both_changes_and_replay_is_a_noop(self):
        db = self.temporary_database()
        engine = make_engine(FixtureModel(), db)
        try:
            result = await engine.ingest_session(
                "s1", [{"role": "user", "content": "周六练琴，周日跑步", "external_id": "turn-1"}]
            )
            assert result["writer_calls"] == 1
            source = engine.store.get_message_ids_by_external(["turn-1"])[0]
            assert sorted(item.title for item in engine.list_items()) == ["练琴", "跑步"]
        finally:
            await engine.close()

        store = LocalMemoryStore(db)
        try:
            operations = [
                {
                    "kind": "add_version", "item_id": item.id, "summary": "再改期",
                    "status": "active", "change_type": "reschedule", "source_quote": "",
                }
                for item in store.list_items()
            ]
            assert store.commit_extraction(source, operations) == []
            assert len(store.list_items()) == 2
            assert [len(item.versions) for item in store.list_items()] == [1, 1]
        finally:
            store.close()


    async def test_raw_only_ingestion_leaves_nothing_for_startup_to_extract(self):
        model = FixtureModel()
        engine = make_engine(model)
        self.addAsyncCleanup(engine.close)
        await engine.ingest_session(
            "s1", [{"content": "原始对话只用于原文对照"}], extract_state=False
        )
        assert await engine.recover_pending_extractions() == 0
        assert model.calls == []
        assert engine.list_items() == []

    def test_failed_batch_rolls_back_both_changes_and_completion_marker(self):
        store = LocalMemoryStore(":memory:")
        try:
            source = store.add_message("s1", "user", "周六练琴，周日跑步")
            operations = [
                {
                    "kind": "create_item", "title": "练琴", "item_kind": "plan",
                    "summary": "周六", "status": "active", "change_type": "create", "source_quote": "",
                },
                {"kind": "create_item"},
            ]
            with self.assertRaisesRegex(KeyError, "title"):
                store.commit_extraction(source.id, operations)
            assert store.list_items() == []
            assert not store.get_message(source.id).extracted
        finally:
            store.close()


    async def test_evidence_receipts_distinguish_complete_partial_and_absent_sources(self):
        for mode in ("oracle", "raw_only", "local_state"):
            with self.subTest(mode=mode):
                await self.check_evidence_receipts(mode)


    async def check_evidence_receipts(self, mode):
        model = FixtureModel()
        engine = make_engine(model)
        try:
            first = engine.store.add_message(
                "bench", "user", "练琴周日" + "甲" * 100,
                namespace="bench", created_at="2026-11-01T00:00:00Z", external_id="gold-a",
            )
            second = engine.store.add_message(
                "bench", "user", "练琴最后的证据",
                namespace="bench", created_at="2026-11-02T00:00:00Z", external_id="gold-b",
            )
            engine.store.add_message(
                "bench", "user", "另一个角色的信息", namespace="other", external_id="gold-a"
            )
            options = {
                "namespace": "bench", "mode": mode, "oracle_external_ids": ["gold-a", "gold-b"]
            }
            result = await engine.evaluate_query("练琴", **options, max_context_bytes=130)
            assert set(result["candidate_message_ids"]) == {first.id, second.id}
            assert result["used_message_ids"] == []
            assert result["used_external_ids"] == []
            assert result["truncated_message_ids"] == [first.id]
            assert result["truncated_external_ids"] == ["gold-a"]
            assert result["omitted_message_ids"] == [second.id]
            assert "练琴周日" in model.context and second.content not in model.context
            assert result["context_bytes"] == len(model.context.encode("utf-8")) <= 130

            enough = await engine.evaluate_query("练琴", **options)
            assert enough["used_message_ids"] == [first.id, second.id]
            assert enough["used_external_ids"] == ["gold-a", "gold-b"]
            assert enough["truncated_message_ids"] == []
            assert enough["omitted_message_ids"] == []
        finally:
            await engine.close()
