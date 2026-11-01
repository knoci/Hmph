"""Shadow framework execution. Cancellation drains native work before quiescence.

Native SDKs can write on synchronous threads. Cancelling an await does not stop
those threads; this worker shields and joins them, with a gate before each batch.
An abruptly killed worker leaves its space unquiesced, never falsely cleaned.
"""

import asyncio
from uuid import uuid4

from hmph.contracts import DomainError
from hmph.memory_build_types import FrameworkBuildError


async def drain(task):
    """Join external work even if the owner is cancelled again during shutdown."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    # Retrieve failures so a retired writer cannot produce an unhandled-task log.
    if not task.cancelled():
        task.exception()


class FrameworkWorker:
    def __init__(self, builds, backend, *, lease_seconds=30, poll_seconds=0.5):
        self.builds, self.backend = builds, backend
        self.lease_seconds, self.poll_seconds = lease_seconds, poll_seconds
        self.worker_id = "framework-" + uuid4().hex

    async def _heartbeat(self, lease):
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            await self.builds.renew(lease, self.lease_seconds)

    async def _cleanup(self, lease):
        work = asyncio.create_task(self.backend.purge(lease))
        cancelled = False
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            pass
        finally:
            await drain(work)
        if work.exception() is None:
            try:
                await self.builds.finish_cleanup(lease, work.result())
            except DomainError:
                await self.builds.fail_cleanup(lease)
        else:
            await self.builds.fail_cleanup(lease)
        if cancelled:
            raise asyncio.CancelledError

    async def run_once(self):
        await self.builds.recover_expired(backend=self.backend.backend)
        cleanup = await self.builds.claim_cleanup(
            self.worker_id, backend=self.backend.backend, lease_seconds=300
        )
        if cleanup is not None:
            await self._cleanup(cleanup)
            return True
        lease = await self.builds.claim(
            self.worker_id, backend=self.backend.backend, lease_seconds=self.lease_seconds
        )
        if lease is None:
            return False
        stopped = asyncio.Event()

        async def checkpoint():
            if stopped.is_set():
                raise DomainError("stale_framework_lease")
            await self.builds.renew(lease, self.lease_seconds)

        work = asyncio.create_task(self.backend.build(lease, checkpoint))
        heartbeat = asyncio.create_task(self._heartbeat(lease))
        success, cancelled = False, False
        error_code, retryable = "build_failed", True
        try:
            done, _ = await asyncio.wait((work, heartbeat), return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                heartbeat.result()
            work.result()
            success = True
        except asyncio.CancelledError:
            cancelled = True
        except FrameworkBuildError as error:
            error_code, retryable = error.code, error.retryable
        except Exception:
            pass
        finally:
            stopped.set()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            await drain(work)
            await self.builds.quiesce(lease, usage=self.backend.usage)
            if success:
                await self.builds.publish(lease)
            else:
                await self.builds.fail(lease, code=error_code, retryable=retryable)
        if cancelled:
            raise asyncio.CancelledError
        return True

    async def run_forever(self):
        while True:
            if not await self.run_once():
                await asyncio.sleep(self.poll_seconds)
