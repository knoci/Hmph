"""Local Mem0 writer supervision. Native algorithms stay in a spawned process.

Each space has a persistent lock inode and a durable execution nonce. A late
child must register under that lock before opening native stores. Recovery holds
the same lock while checking process identity, draining PG sessions and confirming
quiescence. No PID-only kill, timeout-only stop proof, or raw-text spool files.
"""

import asyncio
import hashlib
import json
import multiprocessing
import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import getnode, uuid4

import psutil

from hmph.database import Database
from hmph.memory_build_types import BuildUsage, FrameworkBuildError, WriterIdentity
from hmph.memory_builds import MemoryBuilds


def local_host(root):
    return hashlib.sha256(f"{platform.node()}:{getnode()}:{root}".encode()).hexdigest()


@contextmanager
def execution_lock(root, space_id):
    directory = (root / "artifacts/framework-runtime").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Framework runtime path escaped workspace")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (space_id.hex + ".lock")
    # Never unlink: a delayed child must not acquire a different inode after recovery.
    with path.open("a+b") as handle:
        os.set_inheritable(handle.fileno(), False)
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def stop_identified_process(identity):
    """Only the registered incarnation may be killed; unknown descendants block."""
    # boot_time is diagnostic: Windows values can differ by sub-microseconds
    # between processes and can move with wall-clock adjustments. Never infer
    # death from that comparison; verify the process incarnation itself.
    if identity.pid is None:
        return True
    try:
        process = psutil.Process(identity.pid)
        if process.create_time() != identity.started_at:
            return True  # PID has been reused: never signal its new owner.
        if process.children(recursive=True):
            return False
        process.kill()  # psutil also checks PID reuse before signalling.
        process.wait(timeout=5)
        return True
    except psutil.NoSuchProcess:
        return True
    except (psutil.AccessDenied, psutil.TimeoutExpired):
        return False


def deny_child_processes(event, arguments):
    # The pinned local Mem0 + FastEmbed stack uses threads, not external workers.
    # A provider requiring subprocesses needs a different containment protocol.
    if event in {"subprocess.Popen", "os.fork", "os.forkpty", "os.posix_spawn", "os.system"}:
        raise RuntimeError("Nested native processes are not supported")


async def execute_writer(backend, url, schema, lease, reserved, sender):
    database = Database(url, schema=schema)
    builds = MemoryBuilds(database, digest_key=b"unused-by-framework-writer")
    current = psutil.Process()
    running = reserved.model_copy(update={"pid": current.pid, "started_at": current.create_time()})
    try:
        await builds.start_execution(lease, reserved, running)
        sys.addaudithook(deny_child_processes)

        async def checkpoint():
            # Only the supervisor renews. An orphan cannot keep its own lease alive.
            await builds.validate_execution(lease, running)

        try:
            await backend.build(lease, checkpoint)
            report = {"success": True}
        except FrameworkBuildError as error:
            report = {"success": False, "code": error.code, "retryable": error.retryable}
        except Exception:
            report = {"success": False, "code": "build_failed", "retryable": True}
        report["usage"] = backend.usage.model_dump()
        sender.send_bytes(json.dumps(report).encode())
    finally:
        await database.close()


def writer_main(backend, url, schema, lease, reserved, sender):
    # Native libraries can log provider bodies; none of that becomes process output.
    with open(os.devnull, "w") as sink:
        sys.stdout = sys.stderr = sink
        try:
            with execution_lock(backend.root, lease.space_id) as acquired:
                if acquired:
                    asyncio.run(execute_writer(backend, url, schema, lease, reserved, sender))
        except BaseException:
            pass  # Supervisor records a fixed error and unknown usage if no report arrived.
        finally:
            sender.close()


class ProcessBackend:
    """Execution adapter used by the product CLI, with native Mem0 as delegate."""

    backend = "mem0"

    def __init__(self, builds, native):
        if native.backend != self.backend:
            raise ValueError("Unsupported supervised backend")
        self.builds, self.native = builds, native
        self.root = Path(native.root).resolve()
        self.host_id = local_host(self.root)
        self.usage = None

    async def build(self, lease, checkpoint):
        self.usage = None
        reserved = WriterIdentity(
            host_id=self.host_id, boot_time=psutil.boot_time(), execution_id=uuid4()
        )
        await self.builds.reserve_execution(lease, reserved)
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=writer_main,
            args=(self.native, self.builds.db.url, self.builds.db.schema, lease, reserved, sender),
            daemon=True,
        )
        started = False
        try:
            process.start()
            started = True
            sender.close()
            while process.is_alive():
                await asyncio.sleep(0.2)
                await checkpoint()
            process.join()
            if not receiver.poll():
                raise FrameworkBuildError("build_failed", retryable=True)
            try:
                report = json.loads(receiver.recv_bytes(4096))
                self.usage = BuildUsage.model_validate(report["usage"])
            except (EOFError, OSError, ValueError, KeyError):
                raise FrameworkBuildError("build_failed", retryable=True) from None
            if not report["success"]:
                raise FrameworkBuildError(report["code"], retryable=report["retryable"])
        finally:
            if started:
                if process.is_alive():
                    process.kill()  # Owned live OS handle, not a recovered PID.
                await asyncio.to_thread(process.join)
                process.close()
            sender.close()
            receiver.close()

    async def _confirm(self, lease, identity, *, usage=None):
        if identity.host_id != self.host_id:
            return False
        if not await asyncio.to_thread(stop_identified_process, identity):
            return False
        with execution_lock(self.root, lease.space_id) as acquired:
            if not acquired:
                return False
            await self.native.drain_writes(lease)
            await self.builds.confirm_execution_stopped(lease, identity, usage=usage)
            return True

    async def quiesce(self, lease):
        identity = await self.builds.execution(lease)
        if identity is not None:
            try:
                return await self._confirm(lease, identity, usage=self.usage)
            except Exception:
                await self.builds.stop_unverified(lease)
        return False

    async def recover(self):
        for lease, identity in await self.builds.recoverable_executions(
            backend=self.backend, host_id=self.host_id
        ):
            try:
                if not await self._confirm(lease, identity):
                    await self.builds.stop_unverified(lease)
            except Exception:
                await self.builds.stop_unverified(lease)

    async def purge(self, lease):
        return await self.native.purge(lease)
