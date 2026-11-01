"""Transaction helpers shared by conversation and memory; no network operations."""

import hashlib
import hmac
import json
import re

from sqlalchemy import select

from hmph.contracts import ACTIVE, DomainError
from hmph.dialogue import queue_dialogue
from hmph.models import Job, Owner, StreamEvent, WriteRequest


class RequestLedger:
    def __init__(self, digest_key: bytes):
        self.digest_key = digest_key

    def digest(self, target, body):
        raw = json.dumps([target, body], sort_keys=True, ensure_ascii=False).encode()
        return hmac.new(self.digest_key, raw, hashlib.sha256).hexdigest()

    @staticmethod
    async def owner_lock(session, owner_id):
        if (
            await session.scalar(select(Owner).where(Owner.id == owner_id).with_for_update())
            is None
        ):
            raise DomainError("not_found", 404)

    @staticmethod
    async def request(session, owner_id, operation, request_id, digest):
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id):
            raise DomainError("invalid_request_id", 422)
        existing = await session.get(WriteRequest, (owner_id, operation, request_id))
        if existing and not hmac.compare_digest(existing.digest, digest):
            raise DomainError("idempotency_conflict")
        return existing.result_id if existing else None

    @staticmethod
    def save(session, owner_id, operation, request_id, digest, result_id):
        session.add(
            WriteRequest(
                owner_id=owner_id,
                operation=operation,
                request_id=request_id,
                digest=digest,
                result_id=result_id,
            )
        )


def append_event(session, turn, event_type, payload, *, segment_index=None):
    event = StreamEvent(
        turn_id=turn.id,
        id=turn.next_event_id,
        type=event_type,
        payload=payload,
        segment_index=segment_index,
    )
    turn.next_event_id += 1
    session.add(event)
    return event


async def terminate_turn(session, turn, status, error_code=None):
    if turn.status not in ACTIVE:
        return
    turn.status, turn.error_code = status, error_code
    job = await session.scalar(select(Job).where(Job.turn_id == turn.id).with_for_update())
    job.status, job.lease_until, job.worker_id = "done", None, None
    job.fence += 1
    append_event(session, turn, f"turn.{status}", {"error_code": error_code, "usage": turn.usage})
    await queue_dialogue(session, turn, turn.source_seq + 1)
