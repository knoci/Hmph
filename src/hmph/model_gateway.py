"""The fake adapter proves transport behavior, not model or memory quality."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Protocol

FAKE_SENTENCES = ("哼，收到啦。", "我现在使用的是测试模型，还没有接入真实记忆。")
FALLBACK = "这条回复暂时无法提供，请稍后再试。"


class ModelGateway(Protocol):
    def stream(self, user_content: str) -> AsyncGenerator[str, None]: ...


class FakeModel:
    def __init__(self, *, delay: float = 0.01):
        self.delay = delay

    async def stream(self, user_content: str) -> AsyncGenerator[str, None]:
        for chunk in ("哼，收", "到啦。", "我现在使用的是测试模型，", "还没有接入真实记忆。"):
            await asyncio.sleep(self.delay)
            yield chunk


class FakeOutputGuard:
    """Allow only pre-reviewed fixture sentences. Never use for a real model."""

    async def allows(self, prefix: str, candidate: str) -> bool:
        return candidate in (*FAKE_SENTENCES, FALLBACK)
