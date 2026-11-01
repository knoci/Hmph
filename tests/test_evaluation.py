import json


def test_gold_labels_and_semantic_ids_are_not_exposed_to_memory_or_reader():
    from hmph.evaluation import parse_longmemeval

    raw = {
        "question_id": "secret-abstention-category_abs",
        "question": "我现在在哪里工作？",
        "question_date": "2026/11/01",
        "answer": "GOLD_ONLY_SENTINEL",
        "has_answer": False,
        "answer_session_ids": ["gold-session-id"],
        "question_type": "knowledge-update",
        "haystack_session_ids": ["gold-session-id"],
        "haystack_dates": ["2026/10/31"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "我最近换工作了。",
                    "has_answer": True,
                    "answer": "NESTED_GOLD_SENTINEL",
                }
            ]
        ],
    }
    case = parse_longmemeval(raw)
    exposed = json.dumps(case.model_input().model_dump(), ensure_ascii=False)
    assert "我最近换工作了。" in exposed
    assert "我现在在哪里工作？" in exposed
    for forbidden in (
        "GOLD_ONLY",
        "NESTED_GOLD",
        "secret-abstention",
        "gold-session-id",
        "has_answer",
        "answer_session_ids",
        "question_type",
    ):
        assert forbidden not in exposed
    assert case.gold.answer == "GOLD_ONLY_SENTINEL"
    assert case.gold.evidence_messages == ("s00000/m00000",)


def test_smoke_report_never_claims_fake_accuracy(tmp_path):
    from hmph.evaluation import audit_suite

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            [
                {
                    "question_id": "one",
                    "question": "喜欢什么？",
                    "answer": "茶",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2026/10/31"],
                    "haystack_sessions": [[{"role": "user", "content": "我喜欢茶。"}]],
                }
            ]
        ),
        encoding="utf-8",
    )
    report = audit_suite(suite)
    assert report["input_audit"]["status"] == "passed"
    assert report["case_count"] == 1
    assert report["quality_status"] == "not_run"
    assert report["quality_metrics"] is None
    assert report["inference_requests"] == 0


async def test_real_inference_runner_keeps_gold_out_of_reader_and_reports_failures(tmp_path):
    from hmph.eval_runner import QAReply, run_suite

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            [
                {
                    "question_id": "category-secret_abs",
                    "question": "喜欢什么饮料？",
                    "answer": "GOLD_ONLY_SENTINEL",
                    "answer_session_ids": ["secret-session"],
                    "haystack_session_ids": ["secret-session"],
                    "haystack_dates": ["2026/10/31"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "我喜欢乌龙茶。", "has_answer": True}]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    seen = []

    async def reader(model_input):
        exposed = model_input.model_dump_json()
        for forbidden in ("GOLD_ONLY", "category-secret", "has_answer", "secret-session"):
            assert forbidden not in exposed
        seen.append(model_input)
        if len(seen) == 2:
            raise RuntimeError("PRIVATE_PROVIDER_ERROR_BODY")
        return QAReply(answer="乌龙茶", abstained=False, source_ids=())

    report = await run_suite(suite, reader, profiles=("recent_window", "lexical_raw"), limit=1)
    assert report["inference_requests"] == 2
    assert report["profiles"]["recent_window"]["completed"] == 1
    assert report["profiles"]["lexical_raw"]["failed"] == 1
    assert report["profiles"]["recent_window"]["accuracy"] is None
    assert report["official_score"] is False and report["frontier_status"] == "not_evaluated"
    assert "PRIVATE_PROVIDER" not in json.dumps(report)


def test_dated_duplicate_filler_sessions_are_preserved_without_ambiguous_source_ids():
    from hmph.evaluation import parse_longmemeval

    case = parse_longmemeval(
        {
            "question_id": "case",
            "question": "名字？",
            "answer": "小沐",
            "answer_session_ids": ["evidence"],
            "haystack_session_ids": ["filler", "filler", "evidence"],
            "haystack_dates": ["2026/01/01", "2026/01/02", "2026/01/03"],
            "haystack_sessions": [
                [{"role": "user", "content": "天气很好"}],
                [{"role": "user", "content": "天气很好"}],
                [{"role": "user", "content": "我叫小沐"}],
            ],
        }
    )
    assert len({m.source_id for m in case.history}) == 3
    assert [m.timestamp for m in case.history[:2]] == ["2026/01/01", "2026/01/02"]
    assert case.gold.evidence_sessions == ("s00002",)


async def test_full_history_over_budget_is_unsupported_without_calls_or_fake_zero_accuracy(
    tmp_path,
):
    from hmph.eval_runner import run_suite

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            [
                {
                    "question_id": "large",
                    "question": "What did I say?",
                    "answer": "test",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2026/01/01"],
                    "haystack_sessions": [[{"role": "user", "content": "unrelated filler " * 500}]],
                }
            ]
        ),
        encoding="utf-8",
    )

    async def forbidden_call(value):
        raise AssertionError("No provider call is permitted for unsupported full history")

    report = await run_suite(
        suite,
        forbidden_call,
        judge=forbidden_call,
        profiles=("full_history",),
        limit=1,
        budget_bytes=512,
    )
    summary = report["profiles"]["full_history"]
    assert summary["unsupported"] == 1 and summary["accuracy"] is None
    assert report["inference_requests"] == 0


def test_hybrid_evaluation_ranker_receives_no_gold_and_payload_stays_in_budget():
    from hmph.eval_runner import prepare_input
    from hmph.evaluation import parse_longmemeval

    class SemanticRanker:
        def rank(self, documents, question):
            assert question == "What protective equipment did I buy?"
            assert "GOLD_ONLY" not in str(documents)
            return list(reversed(range(len(documents))))

    case = parse_longmemeval(
        {
            "question_id": "secret_abs",
            "question": "What protective equipment did I buy?",
            "answer": "GOLD_ONLY",
            "haystack_session_ids": ["secret-s1"],
            "haystack_dates": ["2026/01/01"],
            "haystack_sessions": [[{"role": "user", "content": "I bought a K2 helmet."}]],
        }
    )
    payload = prepare_input(case.model_input(), "hybrid_raw", 512, SemanticRanker())
    assert len(payload.model_dump_json().encode()) <= 512
    assert len(payload.sources) == 1 and payload.sources[0].content == "I bought a K2 helmet."


async def test_retrieval_only_reports_evidence_without_inference_or_answer_score(tmp_path):
    from hmph.eval_runner import run_suite

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            [
                {
                    "question_id": "one",
                    "question": "What drink do I like?",
                    "answer": "oolong tea",
                    "answer_session_ids": ["s1"],
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2026/01/01"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "I like to drink oolong tea."}]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    async def forbidden(value):
        raise AssertionError("Retrieval-only must not make provider calls")

    report = await run_suite(
        suite, forbidden, profiles=("lexical_raw",), limit=1, retrieval_only=True
    )
    assert report["inference_requests"] == 0
    assert report["profiles"]["lexical_raw"]["evidence_session_recall"] == 1
    assert report["profiles"]["lexical_raw"]["accuracy"] is None


def test_source_aware_retrieval_does_not_spend_all_slots_on_one_repeated_message():
    from hmph.eval_runner import prepare_input
    from hmph.evaluation import parse_longmemeval

    repeated = "Helmet inspection checks helmet features and helmet maintenance. " * 100
    fact = "My helmet frame cracked after two days; I returned it on Friday."
    case = parse_longmemeval(
        {
            "question_id": "neutral",
            "question": "What happened to my helmet?",
            "answer": "unused gold",
            "haystack_session_ids": ["old", "new"],
            "haystack_dates": ["2026/01/01", "2026/01/02"],
            "haystack_sessions": [
                [{"role": "assistant", "content": repeated}],
                [{"role": "user", "content": fact}],
            ],
        }
    )
    payload = prepare_input(case.model_input(), "source_aware_raw", 12000)
    assert len(payload.sources) == 2
    assert len({s.source_id for s in payload.sources}) == 2
    assert any(s.content == fact and not s.truncated for s in payload.sources)
    for source in payload.sources:
        original = next(m.content for m in case.history if m.source_id == source.source_id)
        assert original[source.start_offset : source.end_offset] == source.content
