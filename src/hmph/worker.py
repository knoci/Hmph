import asyncio
import re
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from uuid import uuid4

from hmph.contracts import DomainError
from hmph.conversation import Conversation
from hmph.deepseek import ProviderError
from hmph.model_gateway import (
    FALLBACK,
    FakeOutputGuard,
    GenerationRequest,
    GuardReview,
    ModelGateway,
    SourceMention,
)


class Worker:
    def __init__(
        self,
        conversation: Conversation,
        model: ModelGateway,
        *,
        lease_seconds: float = 15,
        poll_seconds: float = 0.1,
        model_mode: str = "fake",
        model_name: str = "fake",
        reviewer: Callable[[GenerationRequest, str], Awaitable[GuardReview]] | None = None,
    ):
        if model_mode not in ("fake", "deepseek"):
            raise ValueError("Unknown model mode")
        self.conversation = conversation
        self.model = model
        self.guard = FakeOutputGuard()
        self.worker_id = uuid4().hex
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds
        self.model_mode = model_mode
        self.model_name = model_name
        self.reviewer = reviewer

    async def _generate(self, lease):
        request = await self.conversation.generation_request(lease)
        if self.model_mode == "deepseek":
            try:
                await self._generate_reviewed(lease, request)
            except ProviderError as error:
                if error.code not in {"provider_network", "provider_timeout"}:
                    raise
                # Real replies are buffered before publication. A single
                # network recovery stays in the same fenced user turn.
                await self._generate_reviewed(lease, request, previous_failure=error.code)
            return
        buffer = ""
        prefix = ""
        segment_index = 0
        # Adapters yield async generators; closing also cancels provider resources.
        async with aclosing(self.model.stream(request)) as stream:
            async for delta in stream:
                buffer += delta.text
                while match := re.search(r"[。！？.!?\n]", buffer):
                    candidate, buffer = buffer[: match.end()], buffer[match.end() :]
                    if not await self.guard.allows(prefix, candidate):
                        await self.conversation.publish(lease, segment_index, FALLBACK)
                        await self.conversation.finish(lease)
                        return
                    await self.conversation.publish(lease, segment_index, candidate)
                    prefix += candidate
                    segment_index += 1
                if len(buffer) > 256:
                    await self.conversation.publish(lease, segment_index, FALLBACK)
                    await self.conversation.finish(lease)
                    return
            if buffer:
                candidate = buffer if await self.guard.allows(prefix, buffer) else FALLBACK
                await self.conversation.publish(lease, segment_index, candidate)
            elif segment_index == 0:
                raise DomainError("empty_model_output")
            await self.conversation.finish(lease)

    async def _generate_reviewed(self, lease, request, *, previous_failure=None):
        body = ""
        generation_usage = None
        async with aclosing(self.model.stream(request)) as stream:
            async for delta in stream:
                body += delta.text
                if len(body) > 2000:
                    raise DomainError("output_limit")
                if delta.usage is not None:
                    generation_usage = delta.usage.model_dump()
        if not body.strip():
            raise DomainError("empty_model_output")
        metadata = {
            "provider": "deepseek",
            "model": self.model_name,
            "generation": generation_usage,
            "guard": None,
            "complete": False,
        }
        metadata["complete"] = generation_usage is not None and previous_failure is None
        metadata["generation_attempts"] = 2 if previous_failure else 1
        if previous_failure:
            metadata["previous_generation_error"] = previous_failure
            metadata["previous_generation_usage"] = None
        metadata["guard_action"] = "not_requested"
        mentions = []
        if self.reviewer is not None:
            # Optional source annotation never vetoes character speech. An
            # annotation service outage must not discard a generated reply.
            try:
                review = await self.reviewer(request, body)
                metadata["guard"] = review.usage.model_dump() if review.usage else None
                metadata["complete"] &= review.usage is not None
                metadata["guard_action"] = "annotation_only"
                metadata["guard_reason"] = review.reason
                mentions.extend(review.mentions)
            except Exception as error:
                metadata["guard_action"] = "annotation_unavailable"
                metadata["guard_error_code"] = (
                    error.code if isinstance(error, ProviderError) else "provider_protocol"
                )
                metadata["complete"] = False
        permissions = (
            {d.source_id: d.permission for d in request.companion.memory_use}
            if request.companion
            else {}
        )
        # Verbatim known facts have a deterministic witness too. Paraphrases
        # without a validated shared span are deliberately not guessed into the ledger.
        for hit in request.evidence:
            if 2 <= len(hit.value) <= 80 and hit.value in body:
                for source in hit.sources:
                    if (
                        permissions.get(source.message_id) == "mention"
                        and hit.value in source.quote
                        and len(mentions) < 12
                    ):
                        mentions.append(SourceMention(source_id=source.message_id, quote=hit.value))
        valid_mentions = []
        for hint in mentions:
            reason = await self.conversation.companion_output_reason(lease, body, (hint,))
            if reason == "ok":
                valid_mentions.append(hint)
        metadata["dropped_annotations"] = len(mentions) - len(valid_mentions)
        metadata["policy_reason"] = "ok"
        metadata["policy_stage"] = "source_annotation"
        remaining = body
        mentions = tuple(valid_mentions)
        segment_index = 0
        # Segmentation never adds, replaces or rewrites the model's reply.
        while remaining:
            match = re.search(r"[。！？.!?\n]", remaining[:256])
            length = match.end() if match else min(len(remaining), 256)
            await self.conversation.publish(
                lease, segment_index, remaining[:length], mentions=mentions
            )
            remaining = remaining[length:]
            segment_index += 1
        await self.conversation.finish(lease, usage=metadata)

    async def _heartbeat(self, lease):
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            await self.conversation.renew(lease, self.lease_seconds)

    async def _supervise(self, lease):
        generation = asyncio.create_task(self._generate(lease))
        heartbeat = asyncio.create_task(self._heartbeat(lease))
        tasks = (generation, heartbeat)
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if generation.done():
                return generation.result()
            heartbeat.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run_once(self) -> bool:
        await self.conversation.recover_expired()
        lease = await self.conversation.claim(
            self.worker_id, self.lease_seconds, model_mode=self.model_mode
        )
        if lease is None:
            return False
        try:
            async with asyncio.timeout(lease.remaining_seconds):
                await self._supervise(lease)
        except DomainError as error:
            if error.code != "stale_lease":
                await self._fail(lease, error.code)
            else:
                await self.conversation.recover_expired()
        except TimeoutError:
            await self.conversation.recover_expired()
        except ProviderError as error:
            await self._fail(lease, str(error))
        except Exception:
            # Provider exceptions can contain URLs, credentials and input; do not persist them.
            await self._fail(lease, "model_error")
        return True

    async def _fail(self, lease, code):
        try:
            await self.conversation.finish(lease, error_code=code)
        except DomainError as error:
            if error.code != "stale_lease":
                raise
            await self.conversation.recover_expired()

    async def run_forever(self):
        while True:
            if not await self.run_once():
                await asyncio.sleep(self.poll_seconds)
