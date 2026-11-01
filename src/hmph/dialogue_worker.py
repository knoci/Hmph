"""Bounded background state maintenance; raw delta remains usable during failures."""

import asyncio
from uuid import uuid4

from hmph.contracts import DomainError
from hmph.deepseek import ProviderError
from hmph.dialogue_types import StatePatch


async def fake_update_state(lease):
    return StatePatch(), None


class StateWorker:
    def __init__(self, dialogue, update=fake_update_state, *, model_mode="fake", lease_seconds=30):
        self.dialogue, self.update = dialogue, update
        self.model_mode, self.lease_seconds = model_mode, lease_seconds
        self.worker_id = "state-" + uuid4().hex

    async def _heartbeat(self, lease):
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            await self.dialogue.renew(lease, self.lease_seconds)

    async def run_once(self):
        await self.dialogue.recover_expired()
        lease = await self.dialogue.claim(
            self.worker_id, self.lease_seconds, model_mode=self.model_mode
        )
        if lease is None:
            return False
        if not lease.messages:
            await self.dialogue.commit(lease, StatePatch())
            return True
        update = asyncio.create_task(self.update(lease))
        heartbeat = asyncio.create_task(self._heartbeat(lease))
        usage = None
        try:
            async with asyncio.timeout(45):
                done, _ = await asyncio.wait(
                    (update, heartbeat), return_when=asyncio.FIRST_COMPLETED
                )
                if heartbeat in done:
                    heartbeat.result()
                patch, usage = update.result()
                await self.dialogue.commit(
                    lease, patch, usage=usage.model_dump() if usage else None
                )
        except DomainError as error:
            if error.code != "stale_state_lease":
                await self.dialogue.fail(
                    lease, code="unsupported_state", usage=usage.model_dump() if usage else None
                )
        except ProviderError:
            await self.dialogue.fail(lease, code="provider_failed", retryable=True)
        except TimeoutError:
            await self.dialogue.fail(lease, code="state_timeout", retryable=True)
        except Exception:
            await self.dialogue.fail(lease)
        finally:
            update.cancel()
            heartbeat.cancel()
            await asyncio.gather(update, heartbeat, return_exceptions=True)
        return True

    async def run_forever(self):
        while True:
            if not await self.run_once():
                await asyncio.sleep(0.25)
