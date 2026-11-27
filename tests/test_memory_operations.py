"""Regressions for extraction recovery, SQLite upgrades, and bilingual AIRI use."""

import json
import re
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from test_memory_consistency import make_engine

from hmph.lexical import tokenize
from hmph.local_memory.embeddings import HashingEmbedder
from hmph.local_memory.gateway import create_airi_gateway_app
from hmph.local_memory.llm import InvalidExtraction, OpenAICompatLLM, _parse_turn_extraction
from hmph.local_memory.store import LocalMemoryStore, StaleExtraction


def chat_response(content):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def http_model(handler):
    model = OpenAICompatLLM("fixture", base_url="http://fixture")
    model._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return model


def create_item(title, summary):
    return {
        "kind": "create_item", "title": title, "item_kind": "plan", "summary": summary,
        "status": "active", "change_type": "create", "source_quote": summary,
    }


class ProviderCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_claude_accepts_memory_extraction_and_reply_parameters(self):
        bodies = []

        def respond(request):
            self.assertEqual(str(request.url), "https://api.anthropic.com/v1/chat/completions")
            body = json.loads(request.content)
            bodies.append(body)
            self.assertNotIn("temperature", body)
            self.assertNotIn("top_p", body)
            if body["stream"]:
                return httpx.Response(
                    200, text='data: {"choices":[{"delta":{"content":"Monday"}}]}\n\ndata: [DONE]\n\n',
                    headers={"content-type": "text/event-stream"},
                )
            return chat_response('{"changes":[]}')

        model = OpenAICompatLLM("fixture", base_url="https://api.anthropic.com/v1/", model="claude-fixture")
        model._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(model.aclose)

        question = "When did we reschedule?"
        time_options = {"now_iso": "2026-11-27T09:00:00Z", "timezone_name": "UTC"}
        extraction, _ = await model.extract_changes(
            user_message=question, candidate_items=[], recent_messages=[], **time_options,
        )
        self.assertEqual(extraction.changes, [])
        self.assertNotIn("response_format", bodies[0])

        messages = [{"role": "user", "content": question}]
        await model.proxy_chat_completion(messages=messages, temperature=0.3, extra_params={"top_p": 0.8})
        chunks = [
            chunk async for chunk in model.proxy_chat_completion_stream(
                messages=messages, temperature=0.3, extra_params={"top_p": 0.8},
            )
        ]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "Monday")
        standalone = [
            chunk async for chunk in model.generate_reply_stream(
                user_message=question, recent_messages=[], narrated_context="", **time_options,
            )
        ]
        self.assertEqual(standalone[0]["delta"], "Monday")
        self.assertEqual(len(bodies), 4)


class ExtractionReleaseTests(unittest.IsolatedAsyncioTestCase):
    def test_malformed_output_is_distinct_from_no_changes(self):
        for raw in ("chat", "{}", '{"changes":[', '{"changes":{}}', '{"changes":[{"action":"typo"}]}'):
            with self.subTest(raw=raw), self.assertRaises(InvalidExtraction):
                _parse_turn_extraction(raw)
        self.assertEqual(_parse_turn_extraction('{"changes":[]}').changes, [])

    async def test_structural_failures_survive_restart_and_quarantine_releases_queue(self):
        invalid = "a message with invalid model output"
        later = "I will visit Kyoto on Friday"
        first_responses = iter([chat_response("{}"), httpx.Response(401)])
        db = Path(self.enterContext(TemporaryDirectory(dir="."))) / "hmph.db"
        first = make_engine(http_model(lambda _: next(first_responses)), db)
        try:
            pending = await first.ingest_user_message("s", invalid, namespace="my-role")
            self.assertEqual((pending.extraction_status, pending.extract_attempts), ("pending", 1))
        finally:
            await first.close()

        attempts = []

        def after_restart(request):
            raw = json.loads(json.loads(request.content)["messages"][-1]["content"])["current_user_message"]
            attempts.append(raw)
            if raw == invalid and attempts.count(invalid) < 3:
                return chat_response('{"changes":[')
            changes = [{"action": "create", "title": "Kyoto visit", "summary": later}] if raw == later else []
            return chat_response(json.dumps({"changes": changes}))

        second = make_engine(http_model(after_restart), db)
        self.addAsyncCleanup(second.close)
        app = create_airi_gateway_app(second, gateway_api_key="")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://app") as client:
            healthy = await client.post(
                "/v1/memory/remember", json={"text": later, "session_id": "s", "namespace": "my-role"},
            )
            self.assertEqual(healthy.json()["message"]["extraction_status"], "done")
            self.assertEqual(attempts, [invalid, invalid, later])
            isolated = (await client.get("/v1/memory/extractions?namespace=my-role")).json()["quarantined"]
            self.assertEqual(
                [(m["id"], m["extract_attempts"], m["extraction_failure"]) for m in isolated],
                [(pending.id, 3, "invalid_output")],
            )
            self.assertEqual((await client.get("/v1/memory/extractions")).json()["quarantined"], [])
            retry = await client.post(f"/v1/memory/extractions/{pending.id}/retry?namespace=my-role")
            self.assertEqual(retry.json()["message"]["extraction_status"], "done")
            self.assertEqual([item.title for item in second.list_items(namespace="my-role")], ["Kyoto visit"])

    def test_stale_source_rolls_back_without_closing_the_current_interval(self):
        with closing(LocalMemoryStore(":memory:")) as store:
            base = store.add_message("s", "user", "Saturday")
            old = store.add_message("s", "user", "Sunday")
            new = store.add_message("s", "user", "Monday")
            [item_id] = store.commit_extraction(base.id, [create_item("Piano", "Saturday")])

            def update(summary):
                return {
                    "kind": "add_version", "item_id": item_id, "summary": summary,
                    "status": "active", "change_type": "reschedule", "source_quote": summary,
                }

            store.commit_extraction(new.id, [update("Monday")])
            current = store.get_item(item_id)
            with self.assertRaises(StaleExtraction):
                store.commit_extraction(old.id, [create_item("Second operation", "test"), update("Sunday")])
            self.assertEqual(store.get_message(old.id).extraction_status, "pending")
            self.assertEqual(store.get_item(item_id), current)
            self.assertEqual([v.summary for v in current.versions], ["Saturday", "Monday"])
            self.assertIsNone(current.current.valid_to)
            self.assertEqual(len(store.list_items()), 1)

    async def test_english_and_chinese_memory_in_the_airi_proxy(self):
        for title, create, query, persona, paused, expected in (
            ("Piano practice", "I plan to practice piano on Saturday.", "Is piano practice still on?",
             "You're a witty companion. Reply in English.", "[Paused items.", "Piano practice is paused."),
            ("练琴", "我打算周六练琴。", "练琴计划还在进行吗？", "你是一个俏皮的中文伙伴。",
             "[已暂停/搁置的事项", "练琴已经暂停了。"),
        ):
            with self.subTest(title=title):
                zh = title == "练琴"
                calls = []

                def respond(request, *, calls=calls, title=title, create=create, expected=expected):
                    body = json.loads(request.content)
                    messages = body["messages"]
                    calls.append(messages)
                    if "response_format" in body:
                        change = {"action": "create", "title": title, "summary": create, "source_quote": create}
                        return chat_response(json.dumps({"changes": [change]}, ensure_ascii=False))
                    return chat_response(expected)

                engine = make_engine(http_model(respond))
                try:
                    app = create_airi_gateway_app(engine, gateway_api_key="")
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://app") as client:
                        response = await client.post("/v1/memory/remember", json={"text": create, "session_id": "write"})
                        self.assertEqual(response.json()["item_count"], 1)
                        [item] = engine.list_items()
                        response = await client.post(f"/v1/memory/items/{item.id}/toggle", json={"session_id": "write"})
                        self.assertEqual(response.json()["item"]["status"], "paused")
                        response = await client.post(
                            "/v1/chat/completions",
                            headers={"X-Hmph-Session-Id": "read", "X-Hmph-Memory-Mode": "readonly"},
                            json={"stream": False, "messages": [
                                {"role": "system", "content": persona}, {"role": "user", "content": query},
                            ]},
                        )
                        self.assertEqual(response.json()["choices"][0]["message"]["content"], expected)
                        extraction, proxy = calls
                        self.assertEqual(bool(re.search(r"[\u4e00-\u9fff]", extraction[0]["content"])), zh)
                        system = proxy[0]["content"]
                        self.assertTrue(system.startswith(persona + "\n\n<hmph_memory>"))
                        self.assertIn(paused, system)
                        self.assertIn(create, system)
                        self.assertEqual(bool(re.search(r"[\u4e00-\u9fff]", system)), zh)
                finally:
                    await engine.close()

    async def test_airi_desktop_wrapper_and_notify_payloads_are_sanitized(self):
        forwarded_users = []
        extract_inputs = []

        def respond(request):
            body = json.loads(request.content)
            messages = body["messages"]
            if "response_format" in body:
                payload = json.loads(messages[-1]["content"])
                user_msg = payload["current_user_message"]
                extract_inputs.append(user_msg)
                change = {
                    "action": "create",
                    "title": "Piano practice",
                    "kind": "plan",
                    "summary": "Practice piano on Saturday",
                    "effective_time_text": "周六",
                    "source_quote": user_msg,
                }
                return chat_response(json.dumps({"changes": [change]}, ensure_ascii=False))
            forwarded_users.append(messages[-1]["content"])
            if len(forwarded_users) == 1:
                return chat_response("Pong!")
            return chat_response('<|ACT {"emotion":"happy"}|> 收到')

        engine = make_engine(http_model(respond))
        try:
            app = create_airi_gateway_app(engine, gateway_api_key="")
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://app") as client:
                # 1. Connectivity ping: forwarded upstream, zero DB writes or extraction calls.
                r_ping = await client.post(
                    "/v1/chat/completions",
                    json={"stream": False, "messages": [{"role": "user", "content": "ping"}]},
                )
                self.assertEqual(r_ping.status_code, 200)
                self.assertEqual(r_ping.json()["choices"][0]["message"]["content"], "Pong!")

                # 2. Avatar poke notification + runtime prompt: forwarded upstream, zero DB writes.
                poke_payload = (
                    '{\n  "notify": {\n    "id": "avatar-touch:1",\n    "kind": "ping"\n  }\n}\n\n'
                    '流式控制 token 必须严格使用 `<|NAME payload|>` 形式。'
                )
                r_poke = await client.post(
                    "/v1/chat/completions",
                    json={"stream": False, "messages": [{"role": "user", "content": poke_payload}]},
                )
                self.assertEqual(r_poke.status_code, 200)
                await engine.flush()
                self.assertEqual(engine.get_session_messages("airi-default"), [])
                self.assertEqual(extract_inputs, [])

                # 3. Normal AIRI Desktop utterance with timestamp & [Context] wrapper.
                wrapped_turn = (
                    "[2026-11-27 13:27] 周六练琴\n"
                    "[Context]\n"
                    "- system:airi-runtime-prompt: 流式控制 token ... <|ACT {\"emotion\":\"surprised\"}|> 哇... 你给我准备了礼物吗？"
                )
                r_chat = await client.post(
                    "/v1/chat/completions",
                    json={"stream": False, "messages": [{"role": "user", "content": wrapped_turn}]},
                )
                self.assertEqual(r_chat.status_code, 200)
                await engine.flush()

                # Upstream receives the full original wrapper intact; DB & extraction store ONLY "周六练琴".
                self.assertEqual(forwarded_users[-1], wrapped_turn)
                self.assertEqual(extract_inputs, ["周六练琴"])
                stored_user = [m for m in engine.get_session_messages("airi-default") if m.role == "user"]
                self.assertEqual([m.content for m in stored_user], ["周六练琴"])

                ev = engine.recall_context("练琴约在周几？", session_id="s-new")
                self.assertIn("周六", ev.narrated_context)
                self.assertNotIn("airi-runtime-prompt", ev.narrated_context)
        finally:
            await engine.close()


class TextIndexUpgradeTests(unittest.TestCase):
    def test_unicode_literals_keep_letters_and_cjk_substrings(self):
        tokens = tokenize("ＣＡＦＥ\u0301 mañana Москва ひらがな 한글 中文")
        for token in ("café", "mañana", "москва", "ひら", "らが", "한글", "中文"):
            self.assertIn(token, tokens)

    def test_first_open_rebuilds_four_indexes_and_refreshes_local_hashes_once(self):
        db = Path(self.enterContext(TemporaryDirectory(dir="."))) / "hmph.db"
        embedder = HashingEmbedder()
        phrase = "ひらがな 한글 Москва café"
        tables = ("messages_fts", "items_fts", "session_summaries_fts", "consolidations_fts")
        vectors = ("embeddings", "session_summary_embeddings", "consolidation_embeddings")
        with closing(LocalMemoryStore(db, embedder=embedder)) as store:
            message = store.add_message("s", "user", phrase)
            store.commit_extraction(message.id, [create_item("Travel", phrase)])
            store.upsert_session_summary("s", session_date="2026-11-27", summary=phrase, source_message_ids=[message.id])
            store.replace_consolidations([{"topic": "Travel", "summary": phrase, "source_message_ids": [message.id]}])

        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute("ALTER TABLE messages DROP COLUMN extract_attempts")
            conn.execute("ALTER TABLE messages DROP COLUMN extraction_failure")
            conn.execute("DELETE FROM memory_metadata")
            for table in tables:
                conn.execute(f"UPDATE {table} SET tokens = 'legacy'")
            for table in vectors:
                conn.execute(f"UPDATE {table} SET model_name = ?, vector = zeroblob(length(vector))", (embedder.legacy_name,))

        with closing(LocalMemoryStore(db)) as store:
            self.assertEqual(store.get_message(message.id).content, phrase)
            self.assertEqual(store.get_message(message.id).extraction_status, "done")
        with closing(LocalMemoryStore(db, embedder=embedder)):
            pass
        with closing(sqlite3.connect(db)) as conn:
            for table in tables:
                self.assertEqual(conn.execute(f"SELECT count(*) FROM {table} WHERE {table} MATCH 'ひら'").fetchone()[0], 1)
                self.assertEqual(conn.execute(f"SELECT count(*) FROM {table} WHERE {table} MATCH 'legacy'").fetchone()[0], 0)
            for table, count in zip(vectors, (2, 1, 1), strict=True):
                rows = conn.execute(f"SELECT model_name, vector FROM {table}").fetchall()
                self.assertEqual(len(rows), count)
                for name, vector in rows:
                    self.assertEqual(name, embedder.name)
                    self.assertTrue(any(vector))
        with patch.object(embedder, "embed_passages", side_effect=AssertionError("unexpected reindex")):
            with closing(LocalMemoryStore(db, embedder=embedder)):
                pass

