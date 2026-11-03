import json

import pytest


def trajectory_suite():
    categories = (
        "current_update",
        "late_history",
        "relative_time",
        "same_name",
        "negation_hypothetical",
        "two_hop",
        "abstention",
        "chinese",
    )
    cases = []
    for index, category in enumerate(categories * 2, 1):
        cases.append(
            {
                "id": f"p2-dev-{index:03d}",
                "person_id": f"person-{index}",
                "category": category,
                "build_watermark": 6,
                "conditions": ["cross_session"],
                "history": [
                    {
                        "source_id": f"s{i:03d}",
                        "observed_at": f"2026-10-{i:02d}T10:00:00+08:00",
                        "content": f"Observation {i}: I visited the library.",
                    }
                    for i in range(1, 11)
                ],
                "source_metadata": {
                    f"s{i:03d}": {
                        "session_id": "session-a" if i <= 6 else "session-b",
                        "source_order": i,
                        "source_timezone": "Asia/Hong_Kong",
                    }
                    for i in range(1, 11)
                },
                "query": {"question": "Where did I go?", "timestamp": "2026-10-11T12:00:00+08:00"},
                "gold": {
                    "answer": "library",
                    "aliases": [],
                    "evidence_sources": ["s001"],
                    "rationale": "A single explicit statement is sufficient.",
                },
                "evidence_alternatives": [["s001"], ["s002"]],
            }
        )
    return {
        "schema_version": "hmph-p2-trajectories-v1",
        "split": "development",
        "status": "frozen",
        "cases": cases,
    }


def test_trajectory_writer_stops_at_build_watermark_and_gold_stays_private(tmp_path):
    from hmph.trajectory_eval import load_trajectories, source_context

    path = tmp_path / "suite.json"
    path.write_text(json.dumps(trajectory_suite()), encoding="utf-8")
    cases, _ = load_trajectories(path, require_frozen=True)
    case = cases[0]
    writer = case.writer_input()
    assert len(writer["history"]) == 6
    assert writer["history"][-1]["source_id"] == "s006"
    for label in ("gold", "category", "rationale", "person_id", "evidence_alternatives"):
        assert label not in json.dumps(writer)
    evidence = source_context(case, [{"kind": "source", "source_ids": ["s001"]}])
    assert [item["source_ids"] for item in evidence] == [
        ["s007"],
        ["s008"],
        ["s009"],
        ["s010"],
        ["s001"],
    ]
    assert evidence[-1]["content"] == "Observation 1: I visited the library."


def test_calibration_plan_requires_two_explicit_development_cases_and_hard_caps():
    from hmph.trajectory_eval import RunConfiguration

    config = RunConfiguration.for_protocol("p2a", case_ids=("p2-dev-001", "p2-dev-007"))
    assert config.profiles == ("raw", "graphiti")
    assert config.limits == {"max_calls": 128, "max_input_bytes": 4194304, "per_case_calls": 80}
    assert config.reader_bytes == 5000 and config.evidence_mode == "source_only"
    for changes in (
        {"max_calls": 129},
        {"case_ids": ("p2-holdout-001", "p2-holdout-002")},
        {"evidence_mode": "native"},
        {"profiles": ("raw", "mem0")},
        {"case_ids": ()},
    ):
        with pytest.raises(ValueError):
            RunConfiguration.for_protocol(
                "p2a", **{**config.model_dump(exclude={"protocol"}), **changes}
            )


@pytest.mark.parametrize(
    "mutation", ["future", "duplicate_order", "unknown_source", "draft", "label"]
)
def test_invalid_or_unfrozen_trajectory_cannot_reach_a_writer(tmp_path, mutation):
    from hmph.trajectory_eval import load_trajectories

    data = trajectory_suite()
    case = data["cases"][0]
    if mutation == "future":
        case["history"][-1]["observed_at"] = "2027-01-01T00:00:00+08:00"
    elif mutation == "duplicate_order":
        case["source_metadata"]["s002"]["source_order"] = 1
    elif mutation == "unknown_source":
        case["evidence_alternatives"] = [["s099"]]
    elif mutation == "draft":
        data["status"] = "draft"
    else:
        case["history"][0]["gold"] = "must never reach the writer"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_trajectories(path, require_frozen=True)


def test_alternative_source_is_valid_but_null_with_citation_remains_protocol_failure(tmp_path):
    from dataclasses import replace

    from hmph.selection_eval import Reply
    from hmph.trajectory_eval import grade_trajectory, load_trajectories, source_context

    path = tmp_path / "suite.json"
    path.write_text(json.dumps(trajectory_suite()), encoding="utf-8")
    cases, _ = load_trajectories(path)
    case = cases[0]
    assert grade_trajectory(case, Reply(answer="library", source_ids=("s002",)), {"s002"})[
        "correct"
    ]
    unknown = replace(
        case,
        gold=case.gold.model_copy(update={"answer": None, "evidence_sources": ()}),
        evidence_alternatives=(),
    )
    assert not grade_trajectory(unknown, Reply(answer=None, source_ids=("s002",)), {"s002"})[
        "correct"
    ]
    with pytest.raises(ValueError, match="Unknown native source"):
        source_context(case, [{"kind": "source", "source_ids": ["s099"]}])


def test_gold_audit_input_excludes_author_labels_and_accepts_frozen_alternative_sources():
    import json
    from dataclasses import replace
    from pathlib import Path

    from hmph.selection_eval import Reply
    from hmph.trajectory_eval import load_trajectories
    from scripts.trajectory_audit import audit_input, source_grade

    cases, _ = load_trajectories(Path("evals/fixtures/p2-trajectories-holdout-v2.json"))
    case = cases[2]
    copy = replace(case, gold=case.gold.model_copy(update={"rationale": "PRIVATE_AUTHOR_LABEL"}))
    payload = audit_input(copy).model_dump()
    assert set(payload) == {"history", "query"}
    assert len(payload["history"]) == 10
    assert "PRIVATE_AUTHOR_LABEL" not in json.dumps(payload)
    assert "source_timezone" in payload["history"][0]
    reply = Reply.model_validate_json('{"answer":"Nell","source_ids":["s010"]}')
    assert source_grade(case, reply)["correct"]
