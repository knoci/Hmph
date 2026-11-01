from hmph.config import Settings
from hmph.conversation import Conversation
from hmph.deepseek import DeepSeekModel
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
