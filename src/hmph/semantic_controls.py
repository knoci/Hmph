"""Ground state-updater proposals, then apply them at the next fenced turn claim.

No extra perception call and no second preference store: queued proposals live on
their original turn, while the existing control writers own active state and sources.
"""

import re
from collections import defaultdict

from pydantic import ValidationError
from sqlalchemy import select

from hmph.dialogue_types import ControlCue
from hmph.models import MemoryTask, Message, SharedThread, Turn
from hmph.preference_types import ExpressionPreferences
from hmph.source_policy import retainable

UNSAFE = re.compile(r'如果|假如|例如|他说|她说|朋友|["“”‘’]|\bif\b', re.I)
DURABLE = re.compile(r"以后|今后|从现在|往后|日后|以后都|from now on|going forward", re.I)


def admissible(cue, content):
    if not retainable(content) or len(content) > 400 or UNSAFE.search(content):
        return False
    if cue.kind == "preference":
        if not cue.field or not DURABLE.search(content):
            return False
        try:
            ExpressionPreferences.model_validate({cue.field: cue.value})
        except ValidationError:
            return False
        return cue.field != "nickname" or cue.value is None or cue.value in cue.evidence.quote
    if cue.kind == "boundary":
        return bool(
            cue.target
            and cue.target in cue.evidence.quote
            and cue.mode
            and re.search(r"别|不要|不再|停止", content)
        )
    if cue.kind == "thread":
        return bool(
            cue.operation
            and cue.target
            and cue.target in cue.evidence.quote
            and (cue.operation != "create" or re.search(r"一起|陪我|和我|我们", content))
        )
    return cue.kind == "clarify"


async def stage_controls(session, pair, lease, cues):
    grouped = defaultdict(list)
    for cue in cues:
        source = await session.get(Message, cue.evidence.message_id)
        origin = await session.get(Turn, source.turn_id)
        plan = origin.companion_plan or {}
        if (
            origin.pair_id != pair.id
            or source.role != "user"
            or plan.get("controls_applied")
            or plan.get("thread_control")
            or "semantic_control" in plan
            or not admissible(cue, source.content)
        ):
            continue
        grouped[origin.id].append(cue.model_dump(mode="json"))
    for origin_id, proposals in grouped.items():
        origin = await session.get(Turn, origin_id)
        origin.companion_plan = {
            **(origin.companion_plan or {}),
            "semantic_control": {
                "status": "pending",
                "policy_revision": lease.policy_revision,
                "memory_epoch": lease.memory_epoch,
                "source_revision": (
                    await session.get(Message, proposals[0]["evidence"]["message_id"])
                ).revision,
                "proposals": proposals,
            },
        }


async def consume_controls(session, pair, turn):
    from hmph.companion import apply_controls
    from hmph.shared_threads import apply_thread_command

    rows = list(
        (
            await session.execute(
                select(Turn, Message)
                .join(Message, Message.turn_id == Turn.id)
                .join(MemoryTask, MemoryTask.source_message_id == Message.id)
                .where(
                    Turn.pair_id == pair.id,
                    Turn.id != turn.id,
                    Message.role == "user",
                    Message.deleted.is_(False),
                    Turn.companion_plan["semantic_control"]["status"].astext == "pending",
                )
                .order_by(MemoryTask.source_order)
                .limit(32)
            )
        ).all()
    )
    baseline = pair.policy_revision
    for origin, source in rows:
        record = dict(origin.companion_plan["semantic_control"])
        status = "stale"
        if (
            record["policy_revision"] == baseline
            and record["memory_epoch"] == pair.memory_epoch
            and record["source_revision"] == source.revision
        ):
            cues = [ControlCue.model_validate(c) for c in record["proposals"]]
            cues = [c for c in cues if admissible(c, source.content)]
            patch, boundaries, commands = {}, [], []
            ambiguous = not cues
            for cue in cues:
                if cue.kind == "preference":
                    if cue.field in patch and patch[cue.field] != cue.value:
                        ambiguous = True
                    patch[cue.field] = cue.value
                elif cue.kind == "boundary":
                    boundaries.append(
                        {
                            "target": cue.target,
                            "kind": "nickname" if cue.field == "nickname" else "topic",
                            "mode": cue.mode,
                        }
                    )
                elif cue.kind == "thread":
                    threads = list(
                        await session.scalars(
                            select(SharedThread).where(SharedThread.pair_id == pair.id)
                        )
                    )
                    targets = [t for t in threads if cue.target in t.topic]
                    if cue.operation == "create" and not targets:
                        commands.append(("记下", cue.target))
                    elif len(targets) == 1 and cue.operation != "create":
                        operation = {
                            "pause": "暂停",
                            "resume": "恢复",
                            "resolve": "完成",
                            "drop": "放下",
                            "progress": "进展",
                        }[cue.operation]
                        commands.append((operation, targets[0].topic))
                    else:
                        ambiguous = True
                else:
                    ambiguous = True
            if len(commands) > 1:
                ambiguous = True
            if ambiguous:
                turn.companion_plan = {**(turn.companion_plan or {}), "controls_unclear": True}
                status = "clarify"
            else:
                # Existing capacity/transition limits apply atomically to this proposal.
                from hmph.contracts import DomainError

                try:
                    async with session.begin_nested():
                        if patch or boundaries:
                            await apply_controls(
                                session, pair, turn, source, proposed=(patch, boundaries)
                            )
                        for command in commands:
                            await apply_thread_command(
                                session, pair, turn, source, proposed=command
                            )
                    status = "applied"
                except DomainError:
                    await session.refresh(pair)
                    await session.refresh(turn)
                    turn.companion_plan = {**(turn.companion_plan or {}), "controls_unclear": True}
                    status = "clarify"
        origin.companion_plan = {
            **(origin.companion_plan or {}),
            "semantic_control": {**record, "status": status},
        }
    turn.policy_revision = pair.policy_revision
