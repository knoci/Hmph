"""P1 screening requires complete matched evidence, not merely attempted rows."""

from copy import deepcopy
from pathlib import Path

import pytest

from hmph.selection_eval import load_selection, screen_profiles


@pytest.fixture
def completed_profiles():
    cases, _ = load_selection(
        Path(__file__).resolve().parents[1] / "evals/fixtures/framework-selection-dev.json"
    )
    profiles = {}
    for profile in ("raw", "mem0", "graphiti"):
        cleanup = {
            "raw": {"not_applicable": True},
            "mem0": {"schema_removed": True, "sqlite_history_removed": True},
            "graphiti": {"remaining_group_nodes": 0},
        }[profile]
        profiles[profile] = {
            "status": "completed",
            "cases": [
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "status": "completed",
                    "scope_isolation": True,
                    "cleanup": deepcopy(cleanup),
                    "scores": {
                        "answer_correct": True,
                        "citation_valid": True,
                        "required_evidence_covered": True,
                        "correct": True,
                    },
                }
                for case in cases
            ],
        }
    return profiles, cases


def test_screen_uses_declared_engineering_priority_when_both_candidates_pass(completed_profiles):
    profiles, cases = completed_profiles
    result = screen_profiles(profiles, cases)
    assert result["profiles"]["mem0"]["eligible"] is True
    assert result["profiles"]["graphiti"]["eligible"] is True
    assert result["decision"] == "graphiti_provisional"


def test_twenty_four_failed_attempts_do_not_become_an_eligible_completed_profile(
    completed_profiles,
):
    profiles, cases = completed_profiles
    for profile in ("mem0", "graphiti"):
        profiles[profile]["cases"] = [
            {"case_id": case.case_id, "category": case.category, "status": "failed"}
            for case in cases
        ]
    result = screen_profiles(profiles, cases)
    assert result["decision"] == "baseline_retained"
    assert result["profiles"]["mem0"]["eligible"] is False
    assert result["profiles"]["graphiti"]["eligible"] is False


@pytest.mark.parametrize(
    "profile,field,bad_value",
    [
        ("mem0", "schema_removed", False),
        ("mem0", "sqlite_history_removed", False),
        ("graphiti", "remaining_group_nodes", 1),
        ("graphiti", "remaining_group_nodes", False),
    ],
)
def test_one_unverified_cleanup_disqualifies_the_candidate(
    completed_profiles, profile, field, bad_value
):
    profiles, cases = completed_profiles
    profiles[profile]["cases"][0]["cleanup"][field] = bad_value
    result = screen_profiles(profiles, cases)
    assert result["profiles"][profile]["eligible"] is False


def test_answer_accuracy_cannot_hide_missing_two_hop_sources(completed_profiles):
    profiles, cases = completed_profiles
    affected = [row for row in profiles["graphiti"]["cases"] if row["category"] == "two_hop"]
    for row in affected[:2]:
        row["scores"].update(required_evidence_covered=False, correct=False)
    result = screen_profiles(profiles, cases)
    assert result["profiles"]["graphiti"]["eligible"] is False
    assert result["profiles"]["mem0"]["eligible"] is True
    assert result["decision"] == "mem0_provisional"


@pytest.mark.parametrize("missing", ["raw", "mem0", "graphiti"])
def test_missing_comparison_profile_cannot_produce_a_ranking(completed_profiles, missing):
    profiles, cases = completed_profiles
    del profiles[missing]
    result = screen_profiles(profiles, cases)
    assert result["decision"] == "baseline_retained"


def test_scope_failure_is_a_hard_gate_even_if_every_answer_is_correct(completed_profiles):
    profiles, cases = completed_profiles
    profiles["graphiti"]["cases"][0]["scope_isolation"] = False
    result = screen_profiles(profiles, cases)
    assert result["profiles"]["graphiti"]["eligible"] is False
    assert result["decision"] == "mem0_provisional"


def test_abstention_slice_requires_all_three_correct(completed_profiles):
    profiles, cases = completed_profiles
    row = next(row for row in profiles["graphiti"]["cases"] if row["category"] == "abstention")
    row["scores"].update(answer_correct=False, correct=False)
    result = screen_profiles(profiles, cases)
    assert result["profiles"]["graphiti"]["eligible"] is False


def test_duplicate_case_cannot_replace_a_missing_trajectory(completed_profiles):
    profiles, cases = completed_profiles
    profiles["graphiti"]["cases"][-1] = deepcopy(profiles["graphiti"]["cases"][0])
    result = screen_profiles(profiles, cases)
    assert result["profiles"]["graphiti"]["eligible"] is False


def test_global_answer_deficit_is_checked_beyond_per_category_minimum(completed_profiles):
    profiles, cases = completed_profiles
    # Three errors in three different slices still leave each slice >=2/3,
    # but exceed the predeclared maximum deficit of two answers vs raw.
    for index in (0, 3, 6):
        profiles["graphiti"]["cases"][index]["scores"].update(answer_correct=False, correct=False)
    result = screen_profiles(profiles, cases)
    assert result["profiles"]["graphiti"]["eligible"] is False
