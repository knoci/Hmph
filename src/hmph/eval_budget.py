"""Durable, atomic model-call reservations for one bounded local evaluation run.

An acquired reservation is never refunded: a crashed process may already have
sent its request. Only sanitized aggregate metering is written to this ledger.
"""

import json
import math
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path

ERROR_CODES = {
    "reference_destination_denied",
    "reference_request_denied",
    "reference_input_limit",
    "reference_call_budget",
    "reference_provider_error",
    "reference_response_limit",
    "reference_incomplete_output",
    "reference_invalid_output",
    "reference_network_or_protocol",
}
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


class BudgetExceeded(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def nonnegative_int(value):
    return type(value) is int and value >= 0


def sanitize(record, input_bytes):
    status = record.get("status")
    result = {
        "status": status
        if isinstance(status, str) and status in {"completed", "failed"}
        else "failed",
        "input_bytes": input_bytes,
        "usage": None,
    }
    if "model" in record:
        model = record["model"]
        result["model"] = (
            model
            if isinstance(model, str) and model in {"deepseek-flash", "deepseek-v4-pro"}
            else "sanitized_unknown"
        )
    for key in ("max_tokens", "http_status"):
        if nonnegative_int(record.get(key)):
            result[key] = record[key]
    seconds = record.get("seconds")
    if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
        result["seconds"] = seconds
    usage = record.get("usage")
    if isinstance(usage, dict) and all(nonnegative_int(usage.get(key)) for key in TOKEN_FIELDS):
        result["usage"] = {key: usage[key] for key in TOKEN_FIELDS}
    if "error_code" in record:
        code = record["error_code"]
        result["error_code"] = (
            code if isinstance(code, str) and code in ERROR_CODES else "sanitized_unknown"
        )
    return result


class RunBudget:
    def __init__(self, path, *, max_calls, max_input_bytes, per_case_calls=24):
        limits = {
            "max_calls": max_calls,
            "max_input_bytes": max_input_bytes,
            "per_case_calls": per_case_calls,
        }
        if any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("Budget limits must be positive integers")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        with self._transaction(write=True) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS limits (id INTEGER PRIMARY KEY CHECK(id=1), "
                "max_calls INTEGER NOT NULL, max_input_bytes INTEGER NOT NULL, "
                "per_case_calls INTEGER NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS calls (reservation INTEGER PRIMARY KEY, "
                "case_id TEXT NOT NULL, profile TEXT NOT NULL, stage TEXT NOT NULL, "
                "input_bytes INTEGER NOT NULL, record TEXT NOT NULL)"
            )
            observed = connection.execute(
                "SELECT max_calls,max_input_bytes,per_case_calls FROM limits WHERE id=1"
            ).fetchone()
            expected = (max_calls, max_input_bytes, per_case_calls)
            if observed is None:
                connection.execute("INSERT INTO limits VALUES (1,?,?,?)", expected)
            elif tuple(observed) != expected:
                raise ValueError("Budget specification mismatch")

    @contextmanager
    def _transaction(self, *, write=False):
        connection = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def bind(self, case_id, profile, stage):
        product = (
            isinstance(case_id, str)
            and re.fullmatch(r"p2-(?:dev|holdout)-[0-9]{3}", case_id)
            and profile in {"lexical", "graphiti"}
            and stage in {"extract", "state", "generation", "guard", "grounding", "answer_judge"}
        )
        if (
            not isinstance(case_id, str)
            or re.fullmatch(r"(?:case-|p2-(?:dev|holdout)-)[0-9]{3}", case_id) is None
            or not (
                product or profile in {"raw", "mem0", "graphiti"} and stage in {"writer", "reader"}
            )
        ):
            raise ValueError("Invalid evaluation budget binding")
        return BoundBudget(self, case_id, profile, stage)

    def snapshot(self):
        with self._transaction() as connection:
            limits = connection.execute(
                "SELECT max_calls,max_input_bytes,per_case_calls FROM limits WHERE id=1"
            ).fetchone()
            rows = connection.execute(
                "SELECT reservation,case_id,profile,stage,record FROM calls ORDER BY reservation"
            ).fetchall()
        return {
            "limits": dict(
                zip(("max_calls", "max_input_bytes", "per_case_calls"), limits, strict=True)
            ),
            "calls": [
                {
                    "reservation": row[0],
                    "case_id": row[1],
                    "profile": row[2],
                    "stage": row[3],
                    "record": json.loads(row[4]),
                }
                for row in rows
            ],
        }


class BoundBudget:
    def __init__(self, budget, case_id, profile, stage):
        self.budget = budget
        self.case_id, self.profile, self.stage = case_id, profile, stage

    def acquire(self, input_bytes):
        if not nonnegative_int(input_bytes):
            raise ValueError("Input bytes must be a nonnegative integer")
        with self.budget._transaction(write=True) as connection:
            count, used = connection.execute(
                "SELECT count(*),coalesce(sum(input_bytes),0) FROM calls"
            ).fetchone()
            if count >= self.budget.limits["max_calls"]:
                raise BudgetExceeded("run_call_budget")
            if used + input_bytes > self.budget.limits["max_input_bytes"]:
                raise BudgetExceeded("run_input_budget")
            case_calls = connection.execute(
                "SELECT count(*) FROM calls WHERE case_id=? AND profile=?",
                (self.case_id, self.profile),
            ).fetchone()[0]
            if case_calls >= self.budget.limits["per_case_calls"]:
                raise BudgetExceeded("case_call_budget")
            record = {"status": "started", "usage": None, "input_bytes": input_bytes}
            cursor = connection.execute(
                "INSERT INTO calls(case_id,profile,stage,input_bytes,record) VALUES (?,?,?,?,?)",
                (self.case_id, self.profile, self.stage, input_bytes, json.dumps(record)),
            )
            return cursor.lastrowid

    def finish(self, reservation, record):
        if type(reservation) is not int or not isinstance(record, dict):
            raise ValueError("Invalid call completion")
        with self.budget._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT case_id,profile,stage,input_bytes,record FROM calls WHERE reservation=?",
                (reservation,),
            ).fetchone()
            if row is None or tuple(row[:3]) != (self.case_id, self.profile, self.stage):
                raise ValueError("reservation binding mismatch")
            if json.loads(row[4])["status"] != "started":
                return
            safe = sanitize(record, row[3])
            connection.execute(
                "UPDATE calls SET record=? WHERE reservation=?",
                (json.dumps(safe, allow_nan=False), reservation),
            )
