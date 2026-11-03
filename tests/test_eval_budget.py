import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest


def test_trajectory_author_has_a_separate_bounded_ledger_identity(tmp_path):
    from hmph.eval_budget import RunBudget

    budget = RunBudget(
        tmp_path / "author.sqlite", max_calls=1, max_input_bytes=4096, per_case_calls=1
    )
    bound = budget.bind("p2-holdout-001", "author", "trajectory_author")
    bound.acquire(10)
    row = budget.snapshot()["calls"][0]
    assert row["profile"] == "author" and row["stage"] == "trajectory_author"
    with pytest.raises(ValueError):
        budget.bind("p2-holdout-001", "author", "gold_audit")


def test_p2_trajectory_has_shared_writer_reader_budget_and_rejects_unbounded_ids(tmp_path):
    from hmph.eval_budget import BudgetExceeded, RunBudget

    budget = RunBudget(tmp_path / "p2.sqlite", max_calls=3, max_input_bytes=100, per_case_calls=2)
    budget.bind("p2-dev-001", "graphiti", "writer").acquire(10)
    budget.bind("p2-dev-001", "graphiti", "reader").acquire(10)
    with pytest.raises(BudgetExceeded, match="case_call_budget"):
        budget.bind("p2-dev-001", "graphiti", "reader").acquire(10)
    with pytest.raises(ValueError, match="Invalid evaluation budget binding"):
        budget.bind("arbitrary-personal-content", "graphiti", "writer")


def reserve_in_process(path):
    from hmph.eval_budget import BudgetExceeded, RunBudget

    budget = RunBudget(path, max_calls=8, max_input_bytes=80, per_case_calls=8)
    accepted = 0
    for _ in range(8):
        try:
            budget.bind("case-001", "raw", "reader").acquire(10)
            accepted += 1
        except BudgetExceeded:
            pass
    return accepted


def test_cross_process_reservations_share_one_persistent_hard_limit(tmp_path):
    from hmph.eval_budget import RunBudget

    path = tmp_path / "calls.sqlite"
    budget = RunBudget(path, max_calls=8, max_input_bytes=80, per_case_calls=8)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        accepted = list(pool.map(reserve_in_process, [str(path)] * 4))
    assert sum(accepted) == 8
    snapshot = budget.snapshot()
    assert len(snapshot["calls"]) == 8
    assert sum(call["record"]["input_bytes"] for call in snapshot["calls"]) == 80
    assert all(call["record"]["usage"] is None for call in snapshot["calls"])


def test_thread_races_never_overshoot_input_limit(tmp_path):
    from hmph.eval_budget import BudgetExceeded, RunBudget

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=50, max_input_bytes=25)

    def reserve(_):
        try:
            return budget.bind("case-001", "mem0", "writer").acquire(10)
        except BudgetExceeded as error:
            assert error.code == "run_input_budget"
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert sum(value is not None for value in pool.map(reserve, range(40))) == 2
    assert len(budget.snapshot()["calls"]) == 2


def test_case_budget_counts_writer_and_reader_but_not_other_profiles(tmp_path):
    from hmph.eval_budget import BudgetExceeded, RunBudget

    budget = RunBudget(
        tmp_path / "calls.sqlite", max_calls=10, max_input_bytes=100, per_case_calls=2
    )
    budget.bind("case-001", "mem0", "writer").acquire(10)
    budget.bind("case-001", "mem0", "reader").acquire(10)
    with pytest.raises(BudgetExceeded, match="case_call_budget"):
        budget.bind("case-001", "mem0", "reader").acquire(10)
    budget.bind("case-001", "graphiti", "writer").acquire(10)
    budget.bind("case-002", "mem0", "writer").acquire(10)
    assert len(budget.snapshot()["calls"]) == 4


def test_reopen_preserves_reservations_and_rejects_spec_changes(tmp_path):
    from hmph.eval_budget import RunBudget

    path = tmp_path / "calls.sqlite"
    first = RunBudget(path, max_calls=10, max_input_bytes=100)
    reservation = first.bind("case-001", "raw", "reader").acquire(12)
    second = RunBudget(path, max_calls=10, max_input_bytes=100)
    snapshot = second.snapshot()
    assert snapshot["calls"][0]["reservation"] == reservation
    assert snapshot["calls"][0]["record"]["status"] == "started"
    assert snapshot["calls"][0]["record"]["usage"] is None
    with pytest.raises(ValueError, match="specification mismatch"):
        RunBudget(path, max_calls=11, max_input_bytes=100)


def test_finish_whitelists_fields_and_is_idempotent_without_secret_residue(tmp_path):
    from hmph.eval_budget import RunBudget

    path = tmp_path / "calls.sqlite"
    budget = RunBudget(path, max_calls=3, max_input_bytes=100)
    bound = budget.bind("case-001", "raw", "reader")
    reservation = bound.acquire(12)
    bound.finish(
        reservation,
        {
            "status": "failed",
            "model": "SENSITIVE_SENTINEL",
            "input_bytes": 999,
            "headers": {"Authorization": "SENSITIVE_SENTINEL"},
            "prompt": "SENSITIVE_SENTINEL",
            "error_code": "SENSITIVE_SENTINEL",
            "usage": {"prompt_tokens": 1, "completion_tokens": -1, "total_tokens": 0},
        },
    )
    bound.finish(
        reservation,
        {
            "status": "completed",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        },
    )
    call = budget.snapshot()["calls"][0]["record"]
    assert call["status"] == "failed"
    assert call["input_bytes"] == 12
    assert call["usage"] is None
    assert call["error_code"] == "sanitized_unknown"
    assert "SENSITIVE_SENTINEL" not in json.dumps(budget.snapshot())
    assert b"SENSITIVE_SENTINEL" not in path.read_bytes()


def test_known_usage_is_preserved_and_reservation_cannot_cross_bindings(tmp_path):
    from hmph.eval_budget import RunBudget

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=3, max_input_bytes=100)
    writer = budget.bind("case-001", "graphiti", "writer")
    reservation = writer.acquire(12)
    with pytest.raises(ValueError, match="reservation binding mismatch"):
        budget.bind("case-002", "graphiti", "writer").finish(reservation, {"status": "failed"})
    writer.finish(
        reservation,
        {
            "status": "completed",
            "model": "deepseek-flash",
            "max_tokens": 1024,
            "http_status": 200,
            "seconds": 0.25,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        },
    )
    record = budget.snapshot()["calls"][0]["record"]
    assert record["usage"]["total_tokens"] == 120
    assert record["seconds"] == 0.25
