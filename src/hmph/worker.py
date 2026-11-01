import asyncio
import re
from contextlib import aclosing
from uuid import uuid4

from hmph.contracts import DomainError
from hmph.conversation import Conversation
from hmph.model_gateway import FALLBACK, FakeOutputGuard, ModelGateway


class Worker:
    def __init__(
        self,
        conversation: Conversation,
        model: ModelGateway,
        *,
        lease_seconds: float = 15,
        poll_seconds: float = 0.1,
    ):
        self.conversation = conversation
        self.model = model
        self.guard = FakeOutputGuard()
        self.worker_id = uuid4().hex
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds

    async def _generate(self, lease):
        buffer = ""
        prefix = ""
        segment_index = 0
        # Adapters yield async generators; closing also cancels provider resources.
        async with aclosing(self.model.stream(lease.user_content)) as stream:
            async for chunk in stream:
                buffer += chunk
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
        lease = await self.conversation.claim(self.worker_id, self.lease_seconds)
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
