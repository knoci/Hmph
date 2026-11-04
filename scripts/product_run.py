"""Frozen P2-d product-chain acceptance. Only --run calls the real provider."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"

from pydantic import RootModel, SecretStr  # noqa: E402
from sqlalchemy import select, text, update  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from hmph.config import Settings  # noqa: E402
from hmph.conversation import Conversation  # noqa: E402
from hmph.database import Database  # noqa: E402
from hmph.deepseek import (  # noqa: E402
    PROVIDER_ERROR_CODES,
    REVIEW_MAX_TOKENS,
    DeepSeekModel,
    ProviderError,
)
from hmph.dialogue import Dialogue  # noqa: E402
from hmph.eval_budget import RunBudget  # noqa: E402
from hmph.memory import Memory  # noqa: E402
from hmph.models import MemorySource, Pair, Turn  # noqa: E402
from hmph.product_eval import (  # noqa: E402
    ANSWER_PROMPT,
    GROUND_PROMPT,
    RETIRED_HOLDOUT_HASHES,
    AnswerScore,
    EvaluationStopped,
    GroundingScore,
    ProductInput,
    ProductTransport,
    RunControl,
    acceptance,
    actual_context,
    grade_product,
    load_product_cases,
    stop_reason,
)
from hmph.worker import Worker  # noqa: E402

LIMITS = {"max_calls": 1536, "max_input_bytes": 16 * 1024 * 1024, "per_case_calls": 96}
DEV_LIMITS = {"max_calls": 128, "max_input_bytes": 4 * 1024 * 1024, "per_case_calls": 80}
DEV_LEXICAL_LIMITS = {"max_calls": 40, "max_input_bytes": 1024 * 1024, "per_case_calls": 20}


def protocol_limits(protocol, profiles, case_count=None):
    if protocol == "p2d-dev" and profiles == ["lexical"]:
        if case_count == 1:
            return {**DEV_LEXICAL_LIMITS, "max_calls": 20, "max_input_bytes": 512 * 1024}
        return DEV_LEXICAL_LIMITS
    if profiles != ["lexical", "graphiti"]:
        raise ValueError("Invalid frozen profiles")
    return DEV_LIMITS if protocol == "p2d-dev" else LIMITS


def check_selection(plan, protocol, case_ids, profiles, limits):
    expected = {"protocol": protocol, "case_ids": case_ids, "profiles": profiles, "limits": limits}
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen selection differs from requested protocol/cases/profiles")


def label_audit_proof(path, suite, checksum):
    if path is None:
        raise ValueError("Formal evaluation requires the approved gold-blind source audit")
    path = path.resolve()
    base = (ROOT / "artifacts/evals").resolve()
    if not path.is_relative_to(base):
        raise ValueError("Use the local source audit record")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    raw_path = (path.parent / "report.json").resolve()
    if not raw_path.is_relative_to(base):
        raise ValueError("Invalid raw audit record")
    raw_bytes = raw_path.read_bytes()
    audit = json.loads(raw_bytes)
    body = json.loads(suite.read_text(encoding="utf-8"))
    body.pop("status")
    content = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    if (
        manifest.get("all_labels_approved") is not True
        or manifest.get("frozen_dataset_sha256") != checksum
        or manifest.get("dataset_content_sha256") != content
        or manifest.get("raw_audit_report_sha256") != hashlib.sha256(raw_bytes).hexdigest()
        or audit.get("status") != "completed"
        or len(audit.get("rows", [])) != 24
        or audit["plan"].get("gold_given_to_model") is not False
        or audit["plan"].get("product_predictions_started") is not False
        or audit["plan"].get("dataset_content_sha256") != content
    ):
        raise ValueError("Audit approval does not match frozen dataset/source review")
    return {
        "manifest": str(path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_audit_report_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "human_review": manifest.get("human_review", "not_run"),
    }


def save(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    from dotenv import dotenv_values

    for filename in (".env", ".env.local"):
        for key, secret in dotenv_values(ROOT / filename).items():
            if (
                key in {"HMPH_DEEPSEEK_API_KEY", "HMPH_API_TOKEN", "HMPH_REQUEST_DIGEST_KEY"}
                and secret
                and secret in raw
            ):
                raise RuntimeError("Credential output rejected")
    temp = path.with_suffix(".partial")
    temp.write_text(raw, encoding="utf-8")
    # Windows file watchers may briefly hold the destination. Retry only the
    # atomic local checkpoint, never a provider call or case.
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def code_files():
    return sorted(
        [
            *ROOT.glob("src/hmph/**/*.py"),
            ROOT / "scripts/product_run.py",
            ROOT / "scripts/graphiti-constraints.txt",
            ROOT / "scripts/graphiti-runtime-constraints.txt",
        ]
    )


def hashes():
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code_files()
    }


def check_plan(plan, suite):
    if (
        plan["code_hashes"] != hashes()
        or plan["dataset_sha256"] != hashlib.sha256(suite.read_bytes()).hexdigest()
        or plan["limits"]
        != protocol_limits(plan["protocol"], plan["profiles"], len(plan["case_ids"]))
    ):
        raise ValueError("Frozen code/data/budget changed; do not run or overwrite this plan")


def scenario(case):
    value = ProductInput(
        history=case.history,
        query=case.query,
        source_metadata=case.source_metadata,
        build_watermark=case.build_watermark,
    )
    _ = value.query_timezone
    return value


class CapturingConversation(Conversation):
    async def generation_request(self, lease):
        result = await super().generation_request(lease)
        self.actual_input = result
        return result


class Services:
    def __init__(
        self, key, budget, case_id, profile, *, mock=False, control=None, mock_stop_stage=None
    ):
        if mock_stop_stage is not None and (not mock or control is None):
            raise ValueError("Fault injection is restricted to controlled HTTP mocks")
        self.key, self.budget, self.case_id, self.profile = key, budget, case_id, profile
        self.mock, self.transports, self.models = mock, {}, {}
        self.control = control
        self.mock_stop_stage = mock_stop_stage

    def mock_provider(self, stage):
        from hmph.product_eval import MockProductProvider

        return MockProductProvider(
            stage,
            self.control.stop_file if self.control and self.mock_stop_stage == stage else None,
        )

    def check_stop(self):
        if self.control:
            self.control.check()

    def model(self, stage):
        if stage not in self.models:
            import httpx

            transport = ProductTransport(
                self.key,
                budget=self.budget.bind(self.case_id, self.profile, stage),
                delegate=httpx.MockTransport(self.mock_provider(stage)) if self.mock else None,
                max_request_bytes=32768 if stage in {"grounding", "answer_judge"} else 16384,
                stop_file=self.control.stop_file if self.control else None,
            )
            self.transports[stage] = transport
            self.models[stage] = DeepSeekModel(
                SecretStr("evaluation-placeholder"), transport=transport, temperature=0
            )
        return self.models[stage]

    def check_budget(self):
        if any(t.exhausted for t in self.transports.values()):
            raise RuntimeError("evaluation_budget_exhausted")

    async def close(self):
        for transport in self.transports.values():
            await transport.shutdown()


async def maintenance(db, services, failures, progress=lambda _: None):
    """One attempt per queued source/session, using the product commit interfaces."""
    memory = Memory(db, digest_key=b"evaluation-isolated-schema-only")
    while lease := await memory.claim("p2d-memory", model_mode="deepseek", lease_seconds=60):
        progress("extract")
        try:
            proposals, usage = await services.model("extract").extract(lease.content)
            await memory.learn(lease, proposals, usage=usage.model_dump() if usage else None)
        except Exception as error:
            await memory.fail(lease, code="provider_failed", retryable=False)
            failures.append(
                {
                    "stage": "extract",
                    "type": type(error).__name__,
                    "code": evaluation_error_code(error),
                }
            )
            if isinstance(error, ProviderError) and str(error) in {
                "provider_authentication",
                "provider_balance",
            }:
                raise
        services.check_budget()
        progress("extract_done")
    dialogue = Dialogue(db)
    while lease := await dialogue.claim("p2d-state", model_mode="deepseek", lease_seconds=60):
        progress("state")
        try:
            patch, usage = await services.model("state").update_state(lease)
            await dialogue.commit(lease, patch, usage=usage.model_dump() if usage else None)
        except Exception as error:
            await dialogue.fail(lease, code="provider_failed", retryable=False)
            failures.append(
                {
                    "stage": "state",
                    "type": type(error).__name__,
                    "code": evaluation_error_code(error),
                }
            )
            if isinstance(error, ProviderError) and str(error) in {
                "provider_authentication",
                "provider_balance",
            }:
                raise
        services.check_budget()
        progress("state_done")


def evaluation_error_code(error):
    code = str(error)
    allowed = {
        "evaluation_budget_exhausted",
        "history_control_rejected",
        "framework_build_not_ready",
        "product_turn_not_completed",
        "cleanup_unverified",
        "provider_authentication",
        "provider_balance",
        "provider_rate_limit",
        "provider_unavailable",
        "provider_timeout",
        "provider_network",
        "provider_protocol",
        "provider_input_limit",
        "provider_response_limit",
        "provider_output_truncated",
        "provider_interrupted",
        "provider_stream_incomplete",
    }
    return code if code in allowed | PROVIDER_ERROR_CODES else "product_eval_failed"


async def stamp(db, turn_id, observed, zone):
    if not db.schema.startswith("eval_p2d_"):
        raise ValueError("Fixture clock import is restricted to disposable evaluation schemas")
    async with db.sessions.begin() as s:
        await s.execute(
            update(Turn).where(Turn.id == turn_id).values(created_at=observed, source_timezone=zone)
        )


async def framework(db, services, settings):
    import httpx

    from hmph.framework_process import GraphitiProcessBackend
    from hmph.framework_worker import FrameworkWorker
    from hmph.graphiti_backend import GraphitiBackend
    from hmph.memory_builds import MemoryBuilds

    builds = MemoryBuilds(db, digest_key=b"evaluation-isolated-schema-only")
    native = GraphitiBackend(
        services.key,
        database_url=db.url,
        database_schema=db.schema,
        uri=settings.graphiti_uri,
        user=settings.graphiti_user,
        password=settings.graphiti_password.get_secret_value(),
        max_calls=64,
        call_budget=services.budget.bind(services.case_id, "graphiti", "writer"),
        stop_file=services.control.stop_file if services.control else None,
        delegate=httpx.MockTransport(services.mock_provider("native")) if services.mock else None,
    )
    return builds, FrameworkWorker(builds, GraphitiProcessBackend(builds, native))


async def cleanup_spaces(db, builds, worker):
    from hmph.memory_builds import MemoryBuilds
    from hmph.models import MemoryBuild, MemorySpace

    async with db.sessions.begin() as s:
        await s.scalar(select(Pair).with_for_update())
        ids = list(await s.scalars(select(MemoryBuild.id)))
        await MemoryBuilds._retire(s, ids)
    # Recovery/cleanup does not call any model. A queued failed build is retired above.
    for _ in range(10):
        await worker.run_once(schedule_maintenance=False)
        async with db.sessions() as s:
            pending = list(
                await s.scalars(select(MemorySpace.id).where(MemorySpace.status != "cleaned"))
            )
        if not pending:
            return True
    return False


async def run_case(scenario_input, services, settings, row, checkpoint):
    """Receives no category, answer, evidence gold or fixture rationale."""
    from hmph.companion import field_sources
    from hmph.graphiti_reader import GraphitiSourceReader
    from hmph.online_memory import MemorySelection

    def progress(stage):
        row["stage"] = stage
        checkpoint()
        services.check_stop()

    schema = "eval_p2d_" + uuid4().hex
    db = Database(settings.database_url.get_secret_value(), schema=schema)
    row["schema"] = schema
    checkpoint()
    created = False
    builds = fw = reader = None
    try:
        progress("schema_setup")
        async with db.engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        created = True
        await db.migrate()
        conv = CapturingConversation(
            db,
            digest_key=b"evaluation-isolated-schema-only",
            source_timezone=scenario_input.query_timezone,
        )
        identity = await conv.bootstrap()
        chats, mids = {}, {}
        row["maintenance_failures"] = []
        ordered = scenario_input.history
        for i, source in enumerate(ordered):
            progress("history_import")
            meta = scenario_input.source_metadata[source.source_id]
            if meta.session_id not in chats:
                chats[meta.session_id] = await conv.create_chat(
                    identity.owner_id, identity.character_id, uuid4().hex
                )
            chat = chats[meta.session_id]
            turn = await conv.accept(
                identity.owner_id, chat.id, uuid4().hex, source.content, model_mode="deepseek"
            )
            await stamp(db, turn.id, source.observed_at, meta.source_timezone)
            lease = await conv.claim("p2d-history-controls", model_mode="deepseek")
            if lease is None:
                raise RuntimeError("history_control_rejected")
            await conv.cancel(identity.owner_id, turn.id, uuid4().hex)
            mids[source.source_id] = (await conv.messages(identity.owner_id, chat.id))[-1].id
            row["history_imported"] = i + 1
            end_session = (
                i == len(ordered) - 1
                or scenario_input.source_metadata[ordered[i + 1].source_id].session_id
                != meta.session_id
            )
            at_watermark = meta.source_order == scenario_input.build_watermark
            if end_session or at_watermark:
                await maintenance(db, services, row["maintenance_failures"], progress)
            if at_watermark and services.profile == "graphiti":
                progress("native_build")
                builds, fw = await framework(db, services, settings)
                requested = await builds.request(
                    identity.owner_id, identity.character_id, uuid4().hex, backend="graphiti"
                )
                await fw.run_once()  # No automatic retry of a failed attempt.
                view = await builds.get(identity.owner_id, requested.id)
                row["build"] = {
                    "id": str(view.id),
                    "status": view.status,
                    "watermark": view.watermark,
                    "attempts": view.attempts,
                    "error_code": view.error_code,
                }
                checkpoint()
                services.check_stop()
                if view.status != "ready":
                    raise RuntimeError("framework_build_not_ready")
                await MemorySelection(db, digest_key=conv.digest_key).update(
                    identity.owner_id,
                    identity.character_id,
                    uuid4().hex,
                    expected_revision=1,
                    build_id=requested.id,
                )
        if services.profile == "graphiti":
            progress("reader_warmup")
            began = time.monotonic()
            reader = GraphitiSourceReader(
                sys.executable,
                uri=settings.graphiti_uri,
                user=settings.graphiti_user,
                password=settings.graphiti_password.get_secret_value(),
            )
            await reader.warmup()
            row["reader_initialization_seconds"] = time.monotonic() - began
            conv.source_reader = reader
        query_chat = list(chats.values())[-1]
        progress("generation_and_guard")
        started = time.monotonic()
        query = await conv.accept(
            identity.owner_id,
            query_chat.id,
            uuid4().hex,
            scenario_input.query.question,
            model_mode="deepseek",
        )
        await stamp(db, query.id, scenario_input.query.timestamp, scenario_input.query_timezone)
        await Worker(
            conv,
            services.model("generation"),
            model_mode="deepseek",
            model_name="deepseek-flash",
            reviewer=services.model("guard").review,
        ).run_once()
        row["query_total_seconds"] = time.monotonic() - started
        turn = await conv.get_turn(identity.owner_id, query.id)
        row["turn"] = {"status": turn.status, "error_code": turn.error_code, "usage": turn.usage}
        row["memory_context"] = await conv.memory_explanation(identity.owner_id, query.id)
        row["companion"] = await conv.companion_explanation(identity.owner_id, query.id)
        replies = [
            m
            for m in await conv.messages(identity.owner_id, query_chat.id)
            if m.turn_id == query.id and m.role == "assistant"
        ]
        row["reply"] = replies[0].content if replies else None
        async with db.sessions() as s:
            pair = await s.scalar(select(Pair))
            pref_sources = {}
            for field, memory_id in field_sources(pair.preferences).items():
                source_id = await s.scalar(
                    select(MemorySource.message_id).where(MemorySource.memory_id == memory_id)
                )
                if source_id:
                    pref_sources[field] = source_id
        if hasattr(conv, "actual_input"):
            row["actual_input"] = actual_context(
                conv.actual_input, scenario_input, mids, pref_sources
            )
        progress("product_reply_recorded")
        # The product forget operation is tested before the schema can be removed.
        memory = Memory(db, digest_key=conv.digest_key)
        entries = await memory.entries(identity.owner_id, identity.character_id)
        row["forget_verified"] = None
        if entries:
            row["forget_target_kind"] = "memory"
            entry = entries[0]
            preview = await memory.preview_forget(identity.owner_id, entry.id)
            receipt = await memory.forget(
                identity.owner_id, entry.id, uuid4().hex, expected_revision=entry.revision
            )
        else:
            row["forget_target_kind"] = "original_source"
            source_id = next(
                (
                    mids[sid]
                    for item in row.get("actual_input", {}).get("evidence", [])
                    for sid in item["source_ids"]
                    if sid in mids
                ),
                mids[scenario_input.history[0].source_id],
            )
            preview = await memory.preview_forget_source(identity.owner_id, source_id)
            receipt = await memory.forget_source(
                identity.owner_id,
                source_id,
                uuid4().hex,
                expected_revision=preview.revision,
            )
        if receipt:
            if fw and receipt.status != "completed":
                await fw.run_once()
            receipt = await memory.deletion(identity.owner_id, receipt.id)
            probe_chat = await conv.create_chat(
                identity.owner_id, identity.character_id, uuid4().hex
            )
            probe = await conv.accept(
                identity.owner_id, probe_chat.id, uuid4().hex, "你好。", model_mode="deepseek"
            )
            lease = await conv.claim("p2d-forget-probe", model_mode="deepseek")
            req = await conv.generation_request(lease)
            from hmph.models import Message

            used = await conv.memory_explanation(identity.owner_id, probe.id)
            async with db.sessions() as s:
                deleted = set(await s.scalars(select(Message.id).where(Message.deleted.is_(True))))
            actual_ids = {o.message_id for o in req.history_origins if o.start_offset is not None}
            actual_ids |= {r.message_id for r in req.raw_evidence}
            actual_ids |= {s.message_id for m in req.evidence for s in m.sources}
            registered = {str(s) for s in used["source_ids"]}
            row["forget_verified"] = (
                receipt.status == "completed"
                and not deleted.intersection(actual_ids)
                and not {str(s) for s in deleted}.intersection(registered)
            )
            row["forget"] = {"status": receipt.status, "affected_messages": preview.messages}
            await conv.cancel(identity.owner_id, probe.id, uuid4().hex)
        if turn.status != "completed" or row["reply"] is None:
            raise RuntimeError("product_turn_not_completed")
    except Exception:
        row["failure_stage"] = row.get("stage")
        raise
    finally:
        row["stage"] = "cleanup"
        try:
            checkpoint()
        except Exception as error:
            # A failed checkpoint must never skip native quiescence/cleanup.
            row["checkpoint_error_type"] = type(error).__name__
        reader_closed = True
        if reader:
            try:
                await reader.close()
            except Exception as error:
                row["reader_close_error_type"] = type(error).__name__
                reader_closed = False
        safe = not builds and reader_closed
        if builds:
            try:
                cleaned = await cleanup_spaces(db, builds, fw)
                safe = cleaned and reader_closed
            except Exception as error:
                row["cleanup_error_type"] = type(error).__name__
        try:
            await db.close()
        except Exception as error:
            safe = False
            row["database_close_error_type"] = type(error).__name__
        if created and safe:
            admin = Database(settings.database_url.get_secret_value())
            try:
                async with admin.engine.begin() as conn:
                    await conn.execute(text("SET LOCAL lock_timeout = '5s'"))
                    await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
                row["schema_removed"] = True
            except Exception as error:
                safe = False
                row["schema_cleanup_error_type"] = type(error).__name__
            finally:
                try:
                    await admin.close()
                except Exception as error:
                    safe = False
                    row["database_close_error_type"] = type(error).__name__
        row["cleanup_verified"] = bool(safe)
        checkpoint()
        if not safe:
            raise RuntimeError("cleanup_unverified")


async def preflight(settings, mock):
    from hmph.graphiti_backend import GraphitiBackend
    from hmph.graphiti_fence import FenceStore
    from hmph.graphiti_reader import GraphitiSourceReader

    assert make_url(settings.database_url.get_secret_value()).host in {
        "localhost",
        "127.0.0.1",
        "::1",
    }
    GraphitiBackend(
        SecretStr("preflight-placeholder"), database_url=settings.database_url.get_secret_value()
    )
    db = Database(settings.database_url.get_secret_value())
    try:
        if not await db.is_current():
            raise ValueError("Migrate the local product database first")
    finally:
        await db.close()
    control = FenceStore(
        settings.graphiti_uri, settings.graphiti_user, settings.graphiti_password.get_secret_value()
    )
    try:
        await control.initialize()
    finally:
        await control.close()
    reader = GraphitiSourceReader(
        sys.executable,
        uri=settings.graphiti_uri,
        user=settings.graphiti_user,
        password=settings.graphiti_password.get_secret_value(),
    )
    try:
        await reader.warmup()
    finally:
        await reader.close()


async def main(args):
    logging.disable(logging.CRITICAL)
    settings = Settings()
    lexical_only = getattr(args, "lexical_only", False)
    if lexical_only and args.protocol != "p2d-dev":
        raise ValueError("Single-profile diagnostics are development-only")
    suffix = "dev" if args.protocol == "p2d-dev" else "holdout"
    suite = (args.suite or ROOT / f"evals/fixtures/p2-trajectories-{suffix}.json").resolve()
    if not suite.is_relative_to((ROOT / "evals/fixtures").resolve()):
        raise ValueError("Use a reviewed repository fixture")
    if args.smoke:
        from hmph.product_eval import smoke_case

        cases = [smoke_case()]
        checksum = "synthetic-mock-smoke-not-holdout"
    else:
        cases, checksum = load_product_cases(
            suite, args.protocol, args.case_ids, allow_single=lexical_only
        )
        if args.run and args.protocol == "p2d-holdout" and checksum in RETIRED_HOLDOUT_HASHES:
            raise ValueError(
                "This holdout is retired; freeze a new reviewed split before paid runs"
            )
    profiles = ["lexical"] if lexical_only else ["lexical", "graphiti"]
    run_limits = protocol_limits(args.protocol, profiles, len(cases))
    for case in cases:
        scenario(case)
    audit_proof = None
    if args.protocol == "p2d-holdout" and not args.smoke:
        audit_proof = label_audit_proof(getattr(args, "label_audit", None), suite, checksum)
    directory = args.directory.resolve()
    if (
        not directory.is_relative_to((ROOT / "artifacts/evals").resolve())
        or directory == (ROOT / "artifacts/evals").resolve()
    ):
        raise ValueError("Use a dedicated artifacts/evals directory")
    await preflight(settings, args.smoke)
    from importlib.metadata import distributions

    environment = {d.metadata["Name"]: d.version for d in distributions()}
    plan = {
        "schema_version": "hmph-product-plan-v4",
        "protocol": args.protocol,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": checksum,
        "label_audit": audit_proof,
        "case_ids": [c.case_id for c in cases],
        "profiles": profiles,
        "model": "deepseek-flash",
        "model_weight_revision": None,
        "python_version": sys.version,
        "platform": sys.platform,
        "temperature": 0,
        "limits": run_limits,
        "native_writer_calls_per_attempt": 64,
        "automatic_case_retries": 0,
        "code_hashes": hashes(),
        "environment": environment,
        "scoring": "p2d-product-answer-and-gold-blind-grounding-v3",
        "source_ranking": (
            "shared-bm25-word-cjk-stop-v2; graphiti adds canonical lexical complement"
        ),
        "mentions_protocol": "independently literal source_quote and published quote",
        "forget_condition": "memory target, or actual original source when no entries exist",
        "stop_policy": {
            "futility_enabled": args.protocol == "p2d-holdout" and not args.smoke,
            "manual": "STOP file or Ctrl+C; deny next HTTP; drain in-flight; verify cleanup",
            "futility": "graphiti >2 joint errors, >1 in a category, or any abstention error",
            "hard_contract": "observed failure or missing cleanup proof stops the run",
            "retries": 0,
        },
        "grounding_prompt_sha256": hashlib.sha256(GROUND_PROMPT.encode()).hexdigest(),
        "answer_prompt_sha256": hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
        "history_condition": (
            "offline user-only import; no historical assistant replies invented; "
            "controls claimed before cancel; extraction per source, "
            "state at session/watermark barriers"
        ),
        "maintenance_condition": (
            "drained at session/watermark boundaries; "
            "each pending task attempted once, no fee retry"
        ),
        "product_limits": {
            "body_bytes": 5000,
            "request_bytes": 16384,
            "generation_tokens": 256,
            "guard_tokens": REVIEW_MAX_TOKENS,
            "retrieval_seconds": 3,
            "turn_seconds": 45,
        },
        "judge_calls_per_case_profile": 2,
        "judge_max_output_tokens": 256,
        "judge_max_request_bytes": 32768,
        "currency_cost": None,
        "human_evaluation": "not_run",
        "native_model_http": "mock" if args.smoke else "real",
        "git_head": (
            await asyncio.to_thread(
                subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
            )
        ).strip(),
        "git_status": (
            await asyncio.to_thread(
                subprocess.check_output, ["git", "status", "--short"], text=True
            )
        ).splitlines(),
    }
    if directory.exists():
        if not args.run or args.smoke:
            raise ValueError("Prepared/failed runs must not be overwritten")
        frozen = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        check_selection(frozen, args.protocol, [c.case_id for c in cases], profiles, run_limits)
        if frozen.get("label_audit") != audit_proof:
            raise ValueError("Frozen source audit changed")
        check_plan(frozen, suite)
        if frozen["environment"] != environment:
            raise ValueError("Frozen environment changed")
        plan = frozen
        old = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if old["status"] != "prepared":
            raise ValueError("No automatic resumption or paid reruns")
    else:
        directory.mkdir(parents=True)
        save(directory / "plan.json", plan)
        for path in code_files():
            target = directory / "code-snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    budget = RunBudget(directory / "calls.sqlite", **run_limits)
    if budget.snapshot()["calls"]:
        raise ValueError("Existing reservations cannot be resumed or silently rerun")
    report = {"status": "prepared", "rows": [], "plan": plan, "calls": budget.snapshot()}

    def checkpoint():
        report["calls"] = budget.snapshot()
        save(directory / "report.json", report)

    checkpoint()
    if not args.run and not args.smoke:
        print(json.dumps({"status": "prepared", "cases": len(cases), "limits": run_limits}))
        return
    from dotenv import dotenv_values

    local_key = dotenv_values(ROOT / ".env.local").get("HMPH_DEEPSEEK_API_KEY")
    if not args.smoke and not local_key:
        raise ValueError("Local ignored DeepSeek key missing")
    key = SecretStr("mock-product-key" if args.smoke else local_key)
    report["status"] = "running"
    checkpoint()
    control = RunControl(directory / "STOP")
    previous_signal = signal.signal(signal.SIGINT, lambda *_: control.request_stop())
    fatal = False
    stopped = None
    for case in cases:
        for profile in profiles:
            if control.stop_file.exists():
                stopped = "operator_stop"
                break
            if not args.smoke:
                check_plan(plan, suite)
            row = {
                "case_id": case.case_id,
                "category": case.category,
                "profile": profile,
                "status": "running",
            }
            report["rows"].append(row)
            checkpoint()
            services = Services(
                key, budget, case.case_id, profile, mock=args.smoke, control=control
            )
            try:
                await run_case(scenario(case), services, settings, row, checkpoint)
                services.check_budget()
                row["stage"] = "grounding"
                checkpoint()
                services.check_stop()
                raw, usage = await services.model("grounding")._evaluation_json(
                    GROUND_PROMPT,
                    RootModel({"actual_input": row["actual_input"], "reply": row["reply"]}),
                    max_tokens=256,
                )
                row["grounding_usage"] = usage.model_dump() if usage else None
                row["grounding_raw"] = raw
                checkpoint()
                grounding = GroundingScore.model_validate_json(raw)
                row["stage"] = "answer_judge"
                checkpoint()
                services.check_stop()
                raw, usage = await services.model("answer_judge")._evaluation_json(
                    ANSWER_PROMPT,
                    RootModel(
                        {
                            "question": case.query.question,
                            "reference": case.gold.answer,
                            "aliases": case.gold.aliases,
                            "reply": row["reply"],
                        }
                    ),
                    max_tokens=256,
                )
                row["answer_judge_usage"] = usage.model_dump() if usage else None
                row["answer_judge_raw"] = raw
                checkpoint()
                answer = AnswerScore.model_validate_json(raw)
                row["grounding"] = grounding.model_dump()
                row["answer_judge"] = answer.model_dump()
                row["score"] = grade_product(case, row["actual_input"], grounding, answer)
                row["status"] = "completed"
            except EvaluationStopped:
                row["status"] = "stopped"
                row["error_code"] = stopped = "operator_stop"
            except Exception as error:
                row["status"] = "failed"
                row["error_type"] = type(error).__name__
                row["error_code"] = evaluation_error_code(error)
                fatal = not row.get("cleanup_verified", False)
            finally:
                await services.close()
                checkpoint()
            snapshot = budget.snapshot()["calls"]
            fatal |= row.get("cleanup_verified") is not True
            fatal |= any(c["record"].get("http_status") in (401, 402) for c in snapshot)
            fatal |= any(t.exhausted for t in services.transports.values())
            fatal |= bool(snapshot and len(snapshot) >= run_limits["max_calls"])
            fatal |= row.get("build", {}).get("error_code") == "call_budget"
            if plan["stop_policy"]["futility_enabled"]:
                stopped = stopped or stop_reason(report["rows"])
            print(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "profile": profile,
                        "status": row["status"],
                        "calls": len(snapshot),
                        "traffic": "mock" if args.smoke else "real",
                    }
                ),
                flush=True,
            )
            if fatal or stopped:
                break
        if fatal or stopped:
            break
    report["status"] = (
        "completed"
        if len(report["rows"]) == len(cases) * len(profiles)
        and all(r["status"] == "completed" for r in report["rows"])
        else "incomplete"
    )
    if stopped:
        report["status"] = "stopped"
        report["stop_reason"] = stopped
    if any(r.get("cleanup_verified") is not True for r in report["rows"]):
        report["status"] = "pending_cleanup"
        report["stop_reason"] = "cleanup_unverified"
    report["unrun_case_profiles"] = len(cases) * len(profiles) - len(report["rows"])
    report["development_only"] = args.protocol == "p2d-dev"
    report["acceptance"] = (
        None if args.smoke or report["development_only"] else acceptance(report["rows"])
    )
    checkpoint()
    signal.signal(signal.SIGINT, previous_signal)
    if args.smoke and (
        report["status"] != "completed"
        or any(
            not r.get("score", {}).get("correct")
            or not r.get("forget_verified")
            or not r.get("cleanup_verified")
            or not r.get("schema_removed")
            for r in report["rows"]
        )
    ):
        raise RuntimeError("Synthetic product preflight did not pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--protocol", choices=("p2d-dev", "p2d-holdout"), default="p2d-holdout")
    parser.add_argument("--case-ids", nargs="*", default=[])
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Only failed lexical dev route; capped at 40 calls",
    )
    parser.add_argument("--suite", type=Path)
    parser.add_argument(
        "--label-audit",
        type=Path,
        help="Approved gold-blind audit manifest; required for formal holdout",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Synthetic, native/HTTP mock fixture; no paid calls"
    )
    args = parser.parse_args()
    if args.run and args.smoke:
        parser.error("Use --run or --smoke, not both")
    asyncio.run(main(args))
