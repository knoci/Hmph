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
