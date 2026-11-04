from hmph.config import Settings
from hmph.conversation import Conversation
from hmph.deepseek import DeepSeekModel
from hmph.dialogue import Dialogue
from hmph.dialogue_worker import StateWorker
from hmph.memory import Memory
from hmph.memory_worker import MemoryWorker
from hmph.model_gateway import FakeModel
from hmph.worker import Worker


async def run_local(*, port=8017, stop_file=None):
    """Own the API and the three existing workers in one supervised process."""
    import asyncio

    import uvicorn

    from hmph.api import create_app
    from hmph.database import Database

    if not 0 <= port <= 65535:
        raise ValueError("Invalid local port")
    settings = Settings()
    database = Database(settings.database_url.get_secret_value(), schema=settings.database_schema)
    tasks = []
    server = None
    try:
        conversation = Conversation(
            database,
            digest_key=settings.request_digest_key.get_secret_value().encode(),
            turn_timeout=settings.turn_timeout_seconds,
            source_timezone=settings.source_timezone,
        )
        await conversation.bootstrap()
        app = create_app(settings, conversation=conversation)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                access_log=False,
                proxy_headers=False,
                log_level="warning",
            )
        )
        tasks.append(asyncio.create_task(server.serve(), name="hmph-api"))
        workers = (
            build_worker(conversation, settings),
            build_memory_worker(conversation, settings),
            build_state_worker(conversation, settings),
        )
        tasks.extend(
            asyncio.create_task(w.run_forever(), name=f"hmph-worker-{i}")
            for i, w in enumerate(workers)
        )
        if stop_file is not None:

            async def stop_monitor():
                while not await asyncio.to_thread(stop_file.exists):  # noqa: ASYNC110 - external file
                    await asyncio.sleep(0.2)
                server.should_exit = True

            tasks.append(asyncio.create_task(stop_monitor(), name="hmph-stop"))
        print(f"Hmph: http://127.0.0.1:{port}/ · {settings.model_mode}", flush=True)
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        failed = any(t.get_name().startswith("hmph-worker-") for t in done)
        server.should_exit = True
        if not tasks[0].done():
            try:
                await asyncio.wait_for(asyncio.shield(tasks[0]), timeout=5)
            except TimeoutError:
                pass
        if failed:
            raise RuntimeError("local_worker_stopped")
    finally:
        if server is not None:
            server.should_exit = True
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await database.close()


def build_worker(conversation: Conversation, settings: Settings) -> Worker:
    if settings.model_mode == "fake":
        return Worker(
            conversation,
            FakeModel(delay=settings.fake_delay_seconds),
            lease_seconds=settings.lease_seconds,
        )
    if settings.deepseek_api_key is None:
        raise ValueError("Local DeepSeek credential is not configured")
    model = DeepSeekModel(
        settings.deepseek_api_key,
        model=settings.deepseek_model,
        max_tokens=settings.deepseek_max_tokens,
    )
    return Worker(
        conversation,
        model,
        model_mode="deepseek",
        model_name=settings.deepseek_model,
        reviewer=model.review,
        lease_seconds=settings.lease_seconds,
    )


def build_memory_worker(conversation: Conversation, settings: Settings) -> MemoryWorker:
    memory = Memory(
        conversation.db,
        digest_key=conversation.digest_key,
        source_timezone=conversation.source_timezone,
    )
    if settings.model_mode == "fake":
        return MemoryWorker(memory)
    model = DeepSeekModel(settings.deepseek_api_key, model=settings.deepseek_model)
    return MemoryWorker(memory, model.extract, model_mode="deepseek")


def build_state_worker(conversation: Conversation, settings: Settings) -> StateWorker:
    dialogue = Dialogue(conversation.db)
    if settings.model_mode == "fake":
        return StateWorker(dialogue)
    model = DeepSeekModel(settings.deepseek_api_key, model=settings.deepseek_model)
    return StateWorker(dialogue, model.update_state, model_mode="deepseek")
