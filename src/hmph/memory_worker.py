"""Memory extraction is independent of response latency and generation retries."""

import asyncio
from uuid import uuid4

from hmph.contracts import DomainError
from hmph.deepseek import ProviderError
from hmph.memory import Memory


async def fake_extract(content):
    """No guessed facts and no network in fake mode."""
    return [], None


class MemoryWorker:
    def __init__(
        self,
        memory: Memory,
        extract=fake_extract,
        *,
        model_mode="fake",
        lease_seconds=30,
        poll_seconds=0.25,
    ):
        self.memory, self.extract = memory, extract
        self.model_mode, self.lease_seconds = model_mode, lease_seconds
        self.poll_seconds = poll_seconds
        self.worker_id = "memory-" + uuid4().hex

    async def _heartbeat(self, lease):
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            await self.memory.renew(lease, self.lease_seconds)

    async def run_once(self):
        await self.memory.recover_expired()
        lease = await self.memory.claim(
            self.worker_id, self.lease_seconds, model_mode=self.model_mode
        )
        if lease is None:
            return False
        extraction = asyncio.create_task(self.extract(lease.content))
        heartbeat = asyncio.create_task(self._heartbeat(lease))
        usage = None
        try:
            async with asyncio.timeout(45):
                done, _ = await asyncio.wait(
                    (extraction, heartbeat), return_when=asyncio.FIRST_COMPLETED
                )
                if heartbeat in done:
                    heartbeat.result()
                proposals, usage = extraction.result()
                await self.memory.learn(
                    lease, proposals, usage=usage.model_dump() if usage else None
                )
        except DomainError as error:
            if error.code != "stale_memory_lease":
                await self.memory.fail(
                    lease, code="unsupported_memory", usage=usage.model_dump() if usage else None
                )
        except ProviderError:
            await self.memory.fail(lease, code="provider_failed", retryable=True)
        except TimeoutError:
            await self.memory.fail(lease, code="extraction_timeout", retryable=True)
        except Exception:
            await self.memory.fail(lease)
        finally:
            extraction.cancel()
            heartbeat.cancel()
            await asyncio.gather(extraction, heartbeat, return_exceptions=True)
        return True

    async def run_forever(self):
        while True:
            if not await self.run_once():
                await asyncio.sleep(self.poll_seconds)
