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
        if model_mode not in ("fake", "deepseek") or (
            model_mode == "deepseek" and reviewer is None
        ):
            raise ValueError("Real model execution requires an explicit output reviewer")
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
            await self._generate_reviewed(lease, request)
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

    async def _generate_reviewed(self, lease, request):
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
        try:
            review = await self.reviewer(request, body)
        except ProviderError:
            metadata["guard_action"] = "unavailable_fallback"
            await self.conversation.publish(lease, 0, FALLBACK)
            await self.conversation.finish(lease, error_code="guard_unavailable", usage=metadata)
            return
        metadata["guard"] = review.usage.model_dump() if review.usage is not None else None
        metadata["complete"] = generation_usage is not None and review.usage is not None
        metadata["guard_action"] = "allowed" if review.allowed else "blocked_fallback"
        metadata["guard_reason"] = review.reason
        remaining = body if review.allowed else FALLBACK
        segment_index = 0
        # The full body was reviewed; segmentation never adds or rewrites text.
        while remaining:
            match = re.search(r"[。！？.!?\n]", remaining[:256])
            length = match.end() if match else min(len(remaining), 256)
            await self.conversation.publish(lease, segment_index, remaining[:length])
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
