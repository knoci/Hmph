"""Product evaluation uses real input/output seams, not a second short-answer reader."""

import json
from dataclasses import replace
from importlib.util import find_spec

import httpx
import pytest
from pydantic import SecretStr

from hmph.eval_budget import RunBudget


def test_formal_plan_requires_a_matching_approved_blind_audit():
    import hashlib
    from pathlib import Path

    from scripts.product_run import label_audit_proof

    suite = Path("evals/fixtures/p2-trajectories-holdout-v2.json")
    manifest = Path("artifacts/evals/p2d-holdout-v2-gold-audit-v2/label-adjudication.json")
    checksum = hashlib.sha256(suite.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="requires"):
        label_audit_proof(None, suite, checksum)
    # Local paid audit artifacts are optional in a clean checkout.
    if manifest.exists():
        assert label_audit_proof(manifest, suite, checksum)["manifest_sha256"]
        with pytest.raises(ValueError, match="does not match"):
            label_audit_proof(manifest, suite, "0" * 64)


def test_prepared_plan_rejects_selection_drift_and_bounds_targeted_dev():
    from scripts.product_run import check_selection, protocol_limits

    ids = ["p2-dev-003", "p2-dev-015"]
    profiles = ["lexical"]
    limits = protocol_limits("p2d-dev", profiles)
    assert limits["max_calls"] == 40
    plan = {"protocol": "p2d-dev", "case_ids": ids, "profiles": profiles, "limits": limits}
    check_selection(plan, "p2d-dev", ids, profiles, limits)
    with pytest.raises(ValueError, match="Frozen selection"):
        check_selection(plan, "p2d-dev", ["p2-dev-004", "p2-dev-015"], profiles, limits)
    with pytest.raises(ValueError, match="Frozen selection"):
        check_selection(plan, "p2d-dev", ids, ["lexical", "graphiti"], limits)
    with pytest.raises(ValueError):
        protocol_limits("p2d-holdout", profiles)


@pytest.mark.parametrize(
    "profile",
    [
        "lexical",
        pytest.param(
            "graphiti",
            marks=pytest.mark.skipif(
                find_spec("graphiti_core") is None, reason="native environment"
            ),
        ),
    ],
)
async def test_product_runner_verifies_forgetting_when_extraction_returns_no_entries(
    conversation, tmp_path, profile
):
    from hmph.config import Settings
    from hmph.product_eval import smoke_case
    from scripts.product_run import Services, run_case, scenario

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=30, max_input_bytes=1000000)
    services = Services(SecretStr("mock"), budget, "p2-dev-999", profile, mock=True)
    original = services.mock_provider

    def provider(stage):
        if stage != "extract":
            return original(stage)
        return lambda _: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"proposals":[]}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    services.mock_provider = provider
    row = {}
    try:
        await run_case(
            scenario(smoke_case()),
            services,
            Settings(database_url=conversation.db.url),
            row,
            lambda: None,
        )
    finally:
        await services.close()
    assert row["forget_target_kind"] == "original_source"
    assert row["forget_verified"] and row["cleanup_verified"] and row["schema_removed"]


async def test_checkpoint_failure_cannot_skip_disposable_schema_cleanup(conversation, tmp_path):
    from hmph.config import Settings
    from hmph.product_eval import smoke_case
    from scripts.product_run import Services, run_case, scenario

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=30, max_input_bytes=1000000)
    services = Services(SecretStr("mock"), budget, "p2-dev-999", "lexical", mock=True)
    row = {}

    def checkpoint():
        if row.get("stage") == "cleanup":
            raise OSError("PRIVATE file write failure")

    try:
        with pytest.raises(OSError):
            await run_case(
                scenario(smoke_case()),
                services,
                Settings(database_url=conversation.db.url),
                row,
                checkpoint,
            )
    finally:
        await services.close()
    assert row["cleanup_verified"] and row["schema_removed"]
    assert row["checkpoint_error_type"] == "OSError"


def test_natural_abstention_may_include_separately_grounded_personal_facts():
    from hmph.product_eval import AnswerScore, GroundingScore, grade_product, smoke_case

    case = smoke_case()
    case = replace(case, gold=case.gold.model_copy(update={"answer": None, "evidence_sources": ()}))
    # Natural reply: "Your name is Mira, but you haven't told me your birthday."
    context = {"evidence": [{"source_ids": ["s001"]}]}
    answer = AnswerScore(answer_correct=True, abstains=True, reason="correct")
    grounding = GroundingScore(grounded=True, source_ids=["s001"], reason="supported")
    assert grade_product(case, context, grounding, answer)["correct"]
    invalid = grounding.model_copy(update={"source_ids": ["invented"]})
    assert not grade_product(case, context, invalid, answer)["correct"]


async def test_evaluation_stream_finishes_when_product_finishes_not_when_connection_ends(tmp_path):
    from hmph.deepseek import DeepSeekModel
    from hmph.model_gateway import ChatMessage, GenerationRequest
    from hmph.product_eval import ProductTransport

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Hello."}}]}\n\n'
            yield (
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n'
            )
            yield b"data: [DONE]\n\n"
            raise httpx.ReadTimeout("PRIVATE connection stayed open after completion")

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=1, max_input_bytes=64000)
    transport = ProductTransport(
        SecretStr("mock"),
        budget=budget.bind("p2-dev-999", "lexical", "generation"),
        delegate=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream())),
    )
    model = DeepSeekModel(SecretStr("placeholder"), transport=transport)
    try:
        chunks = [
            c
            async for c in model.stream(
                GenerationRequest(messages=(ChatMessage(role="user", content="Hello"),))
            )
        ]
    finally:
        await transport.shutdown()
    assert "".join(c.text for c in chunks) == "Hello."
    record = budget.snapshot()["calls"][0]["record"]
    assert record["status"] == "completed" and record["usage"]["total_tokens"] == 12


@pytest.mark.parametrize(
    "shape,error,record_code",
    [
        ("truncated", "provider_stream_incomplete", "reference_incomplete_output"),
        ("timeout_after_usage", "provider_timeout", "reference_timeout"),
        ("malformed", "provider_protocol", "reference_protocol"),
        ("gzip", None, None),
    ],
)
async def test_product_stream_failure_metering_and_decoding(tmp_path, shape, error, record_code):
    import gzip

    from hmph.deepseek import DeepSeekModel, ProviderError
    from hmph.model_gateway import ChatMessage, GenerationRequest
    from hmph.product_eval import ProductTransport

    end = (
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n'
    )
    data = b'data: {"choices":[{"delta":{"content":"Hello."}}]}\n\n' + end
    if shape == "malformed":
        data = b"data: PRIVATE invalid JSON\n\n"
    if shape == "gzip":
        data = gzip.compress(data + b"data: [DONE]\n\n")

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield data
            if shape == "timeout_after_usage":
                raise httpx.ReadTimeout("PRIVATE body")

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=1, max_input_bytes=64000)
    transport = ProductTransport(
        SecretStr("mock"),
        budget=budget.bind("p2-dev-999", "lexical", "generation"),
        delegate=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                stream=Stream(),
                headers={"content-encoding": "gzip"} if shape == "gzip" else {},
            )
        ),
    )
    model = DeepSeekModel(SecretStr("placeholder"), transport=transport)

    async def consume():
        return [
            c
            async for c in model.stream(
                GenerationRequest(messages=(ChatMessage(role="user", content="Hi"),))
            )
        ]

    try:
        if error:
            with pytest.raises(ProviderError, match=error):
                await consume()
        else:
            assert "".join(c.text for c in await consume()) == "Hello."
    finally:
        await transport.shutdown()
    record = budget.snapshot()["calls"][0]["record"]
    assert record["status"] == ("failed" if error else "completed")
    assert record.get("error_code") == record_code
    assert (record["usage"]["total_tokens"] if record["usage"] else None) == (
        None if shape == "malformed" else 12
    )
    assert "PRIVATE" not in json.dumps(record)


def test_development_protocol_requires_two_explicit_cases_and_has_no_holdout_labels():
    from pathlib import Path

    from hmph.product_eval import load_product_cases

    path = Path("evals/fixtures/p2-trajectories-dev.json")
    ids = ("p2-dev-003", "p2-dev-015")
    cases, checksum = load_product_cases(path, "p2d-dev", ids)
    assert [c.case_id for c in cases] == list(ids) and len(checksum) == 64
    for bad in ((), ids[:1], (*ids, "p2-dev-016"), (ids[0], ids[0]), ("p2-dev-003", "p2-dev-999")):
        with pytest.raises(ValueError):
            load_product_cases(path, "p2d-dev", bad)
    with pytest.raises(ValueError):
        load_product_cases(path, "p2d-holdout", ())


async def test_retired_holdout_is_rejected_before_preflight_or_creating_run(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("retired_runner", Path("scripts/product_run.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    async def preflight(*args):
        pytest.fail("Retired paid run must be rejected before contacting services")

    monkeypatch.setattr(runner, "preflight", preflight)
    output = tmp_path / "retired"
    with pytest.raises(ValueError, match="retired"):
        await runner.main(
            SimpleNamespace(
                protocol="p2d-holdout",
                suite=None,
                case_ids=[],
                directory=output,
                run=True,
                smoke=False,
            )
        )
    assert not output.exists()


@pytest.mark.parametrize("kind", ["native", "product"])
async def test_stop_between_reservation_and_send_never_calls_provider(tmp_path, kind):
    from hmph.product_eval import ProductTransport, RunControl
    from hmph.reference_transport import ReferenceCallError, ReferenceTransport

    control = RunControl(tmp_path / "STOP")
    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=2, max_input_bytes=64000)
    bound = budget.bind("p2-dev-999", "graphiti", "writer" if kind == "native" else "guard")

    class StopOnAcquire:
        def acquire(self, size):
            reserved = bound.acquire(size)
            control.request_stop()
            return reserved

        def finish(self, reservation, record):
            return bound.finish(reservation, record)

    def endpoint(request):
        pytest.fail("Provider must not be contacted after STOP")

    cls = ReferenceTransport if kind == "native" else ProductTransport
    transport = cls(
        SecretStr("placeholder"),
        delegate=httpx.MockTransport(endpoint),
        budget=StopOnAcquire(),
        stop_file=control.stop_file,
    )
    request = httpx.Request(
        "POST", "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
    )
    try:
        with pytest.raises(ReferenceCallError, match="reference_cancelled"):
            if kind == "native":
                transport.handle_request(request)
            else:
                await transport.handle_async_request(request)
    finally:
        if kind == "native":
            transport.close()
        else:
            await transport.shutdown()
    assert len(budget.snapshot()["calls"]) == 1
    assert budget.snapshot()["calls"][0]["record"]["error_code"] == "reference_cancelled"


async def test_unproved_cleanup_cannot_proceed_to_judging(conversation, tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    from sqlalchemy import text

    from hmph.config import Settings
    from hmph.database import Database
    from hmph.product_eval import smoke_case

    spec = importlib.util.spec_from_file_location("cleanup_runner", Path("scripts/product_run.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=30, max_input_bytes=1000000)
    services = runner.Services(SecretStr("mock"), budget, "p2-dev-999", "lexical", mock=True)
    original = Database.close

    async def refused_close(db):
        if db.schema.startswith("eval_p2d_"):
            await original(db)
            raise RuntimeError("PRIVATE close failure")
        await original(db)

    monkeypatch.setattr(Database, "close", refused_close)
    row = {}
    try:
        with pytest.raises(RuntimeError, match="cleanup_unverified"):
            await runner.run_case(
                runner.scenario(smoke_case()),
                services,
                Settings(database_url=conversation.db.url),
                row,
                lambda: None,
            )
        assert row["cleanup_verified"] is False and not row.get("schema_removed")
        assert all(
            c["stage"] not in {"grounding", "answer_judge"} for c in budget.snapshot()["calls"]
        )
    finally:
        await services.close()
        # The injected failure has been resolved by the test; its schema may now be removed.
        monkeypatch.setattr(Database, "close", original)
        async with conversation.db.engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{row["schema"]}" CASCADE'))


@pytest.mark.parametrize("kind", ["native", "product"])
async def test_manual_stop_denies_next_http_without_refunding_inflight(tmp_path, kind):
    from hmph.product_eval import EvaluationStopped, ProductTransport, RunControl
    from hmph.reference_transport import ReferenceCallError, ReferenceTransport

    control = RunControl(tmp_path / "STOP")
    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=5, max_input_bytes=64000)
    requests = []

    def endpoint(request):
        requests.append(request)
        control.request_stop()
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    cls = ReferenceTransport if kind == "native" else ProductTransport
    transport = cls(
        SecretStr("local-placeholder"),
        delegate=httpx.MockTransport(endpoint),
        budget=budget.bind("p2-dev-999", "graphiti", "writer" if kind == "native" else "guard"),
        stop_file=control.stop_file,
    )
    request = httpx.Request(
        "POST", "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
    )
    if kind == "native":
        transport.handle_request(request).close()
    else:
        await transport.handle_async_request(request)
    with pytest.raises(ReferenceCallError, match="reference_cancelled"):
        if kind == "native":
            transport.handle_request(request)
        else:
            await transport.handle_async_request(request)
    with pytest.raises(EvaluationStopped, match="operator_stop"):
        control.check()
    assert len(requests) == len(budget.snapshot()["calls"]) == 1
    assert budget.snapshot()["calls"][0]["record"]["usage"]["total_tokens"] == 12
    if kind == "native":
        transport.close()
    else:
        await transport.shutdown()


def test_futility_stops_at_frozen_thresholds_and_missing_contract_is_not_observed_failure():
    from hmph.product_eval import acceptance, stop_reason

    rows = [
        {
            "profile": "graphiti",
            "case_id": f"p2-holdout-{i:03d}",
            "category": category,
            "status": "failed",
            "cleanup_verified": True,
        }
        for i, category in enumerate(("current_update", "late_history", "relative_time"), 1)
    ]
    assert stop_reason(rows[:2]) is None
    assert stop_reason(rows) == "quality_ceiling"
    assert stop_reason([rows[0], {**rows[1], "category": "current_update"}]) == "category_ceiling"
    assert stop_reason([{**rows[0], "category": "abstention"}]) == "abstention_failure"
    assert stop_reason([{**rows[0], "forget_verified": False}]) == "hard_contract_failure"
    summary = acceptance(rows)
    assert not summary["passed"]
    assert summary["profiles"]["graphiti"]["hard_contract_failures"] == 0
    assert summary["profiles"]["graphiti"]["hard_contract_missing"] == 3


async def test_product_stop_during_maintenance_checkpoints_and_cleans_schema(
    conversation, tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path

    from hmph.config import Settings
    from hmph.product_eval import EvaluationStopped, MockProductProvider, RunControl, smoke_case

    spec = importlib.util.spec_from_file_location("product_runner", Path("scripts/product_run.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    control = RunControl(tmp_path / "STOP")
    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=30, max_input_bytes=100000)
    original = MockProductProvider.__call__

    def endpoint(self, request):
        if self.stage == "extract":
            control.request_stop()
        return original(self, request)

    monkeypatch.setattr(MockProductProvider, "__call__", endpoint)
    services = runner.Services(
        SecretStr("mock"), budget, "p2-dev-999", "lexical", mock=True, control=control
    )
    row, stages = {}, []
    settings = Settings(database_url=conversation.db.url)
    try:
        with pytest.raises(EvaluationStopped):
            await runner.run_case(
                runner.scenario(smoke_case()),
                services,
                settings,
                row,
                lambda: stages.append(row.get("stage")),
            )
    finally:
        await services.close()
    assert "extract" in stages
    assert row["cleanup_verified"] and row["schema_removed"]
    assert len(budget.snapshot()["calls"]) == 1


async def test_product_stream_is_metered_without_changing_the_sse_body(tmp_path):
    from hmph.product_eval import ProductTransport

    budget = RunBudget(
        tmp_path / "calls.sqlite", max_calls=2, max_input_bytes=4096, per_case_calls=2
    )
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    stream = "data: " + json.dumps({"choices": [{"delta": {"content": "你好。"}}]}) + "\n\n"
    stream += (
        "data: "
        + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage})
        + "\n\ndata: [DONE]\n\n"
    )

    def endpoint(request):
        assert request.headers["Authorization"] == "Bearer local-test-not-a-real-key"
        return httpx.Response(200, text=stream)

    transport = ProductTransport(
        SecretStr("local-test-not-a-real-key"),
        budget=budget.bind("p2-dev-001", "lexical", "generation"),
        delegate=httpx.MockTransport(endpoint),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        result = await client.post(
            "https://api.deepseek.com/chat/completions",
            json={
                "model": "deepseek-flash",
                "stream": True,
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 256,
            },
        )
    assert result.text == stream
    record = budget.snapshot()["calls"][0]["record"]
    assert record["usage"] == usage and record["status"] == "completed"
    assert "local-test-not-a-real-key" not in json.dumps(budget.snapshot())


async def test_product_context_records_exact_history_origins_and_current_reference_time(
    conversation,
):
    identity = await conversation.bootstrap()
    chat = await conversation.create_chat(identity.owner_id, identity.character_id, "chat")
    first = await conversation.accept(identity.owner_id, chat.id, "old", "我喜欢跑步。")
    await conversation.cancel(identity.owner_id, first.id, "cancel")
    await conversation.accept(identity.owner_id, chat.id, "query", "记得我喜欢什么运动吗？")
    request = await conversation.generation_request(await conversation.claim("eval"))
    assert request.reference_time.tzinfo is not None
    stored = {m.id: m for m in await conversation.messages(identity.owner_id, chat.id)}
    assert len(request.history_origins) == len(request.messages)
    for message, origin in zip(request.messages, request.history_origins, strict=True):
        assert (
            stored[origin.message_id].content[origin.start_offset : origin.end_offset]
            == message.content
        )


def test_product_scores_keep_available_evidence_separate_from_supported_answer():
    from hmph.product_eval import AnswerScore, GroundingScore, grade_product, smoke_case

    case = smoke_case()
    context = {"evidence": [{"source_ids": ["s001"]}, {"source_ids": ["s002"]}]}
    correct = AnswerScore(answer_correct=True, abstains=False, reason="correct")
    # A correct answer with the wrong actual support must fail the joint gate.
    grounded = GroundingScore(grounded=True, source_ids=["s002"], reason="supported")
    score = grade_product(case, context, grounded, correct)
    assert score["answer_correct"] and score["context_required_evidence_covered"]
    assert not score["answer_evidence_supported"] and not score["correct"]
    grounded = grounded.model_copy(update={"source_ids": ["s001", "invented"]})
    score = grade_product(case, context, grounded, correct)
    assert not score["grounding_judge_source_ids_valid"] and not score["correct"]


def test_product_abstention_requires_no_factual_claim_and_empty_references():
    from hmph.product_eval import AnswerScore, GroundingScore, grade_product, smoke_case

    case = smoke_case()
    case = replace(case, gold=case.gold.model_copy(update={"answer": None, "evidence_sources": ()}))
    context = {"evidence": [{"source_ids": ["s001"]}]}
    answer = AnswerScore(answer_correct=True, abstains=True, reason="correct")
    grounded = GroundingScore(grounded=True, source_ids=[], reason="no_factual_answer")
    assert grade_product(case, context, grounded, answer)["correct"]
    with pytest.raises(ValueError):
        GroundingScore(grounded=True, source_ids=["s001"], reason="no_factual_answer")
    with pytest.raises(ValueError):
        GroundingScore(grounded=True, source_ids=[], reason="unsupported")
    with pytest.raises(ValueError):
        AnswerScore(answer_correct=True, abstains=False, reason="wrong")


def test_product_input_cannot_carry_gold_or_category():
    from hmph.product_eval import ProductInput, smoke_case

    case = smoke_case()
    value = dict(
        history=case.history,
        query=case.query,
        source_metadata=case.source_metadata,
        build_watermark=case.build_watermark,
    )
    assert ProductInput(**value).query_timezone == "UTC"
    with pytest.raises(ValueError):
        ProductInput(**value, gold=case.gold)
    with pytest.raises(ValueError):
        ProductInput(**value, category=case.category)


async def test_product_provider_failure_spends_reservation_and_cannot_retry_over_budget(tmp_path):
    from hmph.product_eval import ProductTransport
    from hmph.reference_transport import ReferenceCallError

    requests = []

    def endpoint(request):
        requests.append(request)
        return httpx.Response(402, text="sensitive provider error must not enter ledger")

    budget = RunBudget(tmp_path / "calls.sqlite", max_calls=1, max_input_bytes=4096)
    transport = ProductTransport(
        SecretStr("local-test"),
        budget=budget.bind("p2-dev-999", "lexical", "guard"),
        delegate=httpx.MockTransport(endpoint),
    )
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.post(
            "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
        )
        assert response.status_code == 402
        with pytest.raises(ReferenceCallError):
            await client.post(
                "https://api.deepseek.com/chat/completions", json={"model": "deepseek-flash"}
            )
    assert len(requests) == 1 and transport.exhausted
    snapshot = budget.snapshot()
    assert snapshot["calls"][0]["record"]["http_status"] == 402
    assert "sensitive provider" not in json.dumps(snapshot)
    await transport.shutdown()


def test_grounding_cannot_see_dates_absent_from_actual_history_prompt():
    from uuid import uuid4

    from hmph.model_gateway import ChatMessage, ContextOrigin, GenerationRequest
    from hmph.product_eval import ProductInput, actual_context, smoke_case

    case = smoke_case()
    mid = uuid4()
    content = case.history[0].content
    request = GenerationRequest(
        messages=(ChatMessage(role="user", content=content),),
        history_origins=(
            ContextOrigin(
                message_id=mid, message_revision=1, start_offset=0, end_offset=len(content)
            ),
        ),
        reference_time=case.query.timestamp,
    )
    scenario = ProductInput(
        history=case.history,
        query=case.query,
        source_metadata=case.source_metadata,
        build_watermark=case.build_watermark,
    )
    context = actual_context(request, scenario, {"s001": mid}, {})
    assert context["evidence"] == [
        {"source_ids": ["s001"], "content": content, "channel": "history"}
    ]
    assert str(mid) not in json.dumps(context)
    assert case.history[0].observed_at.isoformat() not in json.dumps(context)
