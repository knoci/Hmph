from hmph.config import Settings
from hmph.conversation import Conversation
from hmph.deepseek import DeepSeekModel
from hmph.dialogue import Dialogue
from hmph.dialogue_worker import StateWorker
from hmph.memory import Memory
from hmph.memory_worker import MemoryWorker
from hmph.model_gateway import FakeModel
from hmph.worker import Worker


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
