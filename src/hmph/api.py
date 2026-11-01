import asyncio
import hmac
import time
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy import text

from hmph.config import Settings
from hmph.contracts import TERMINAL, DomainError
from hmph.conversation import DEFAULT_CHARACTER, Conversation
from hmph.database import Database


class WriteBody(BaseModel):
    request_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


class CreateChat(WriteBody):
    character_id: UUID = DEFAULT_CHARACTER


class CreateTurn(WriteBody):
    content: str = Field(min_length=1, max_length=4000)


def create_app(settings: Settings | None = None, *, conversation: Conversation | None = None):
    settings = settings or Settings()
    managed = conversation is None
    if managed:
        database = Database(
            settings.database_url.get_secret_value(), schema=settings.database_schema
        )
        conversation = Conversation(
            database,
            digest_key=settings.request_digest_key.get_secret_value().encode(),
            turn_timeout=settings.turn_timeout_seconds,
        )

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.identity = await conversation.bootstrap()
            yield
        finally:
            if managed:
                await conversation.db.close()

    app = FastAPI(title="Hmph", version="0.1.0", lifespan=lifespan)
    security = HTTPBearer(auto_error=False)

    async def require_owner(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    ):
        if credentials is None or not hmac.compare_digest(
            credentials.credentials.encode(),
            settings.api_token.get_secret_value().encode(),
        ):
            raise DomainError("unauthorized", 401)
        return app.state.identity.owner_id

    @app.exception_handler(DomainError)
    async def domain_error(request, error):
        return JSONResponse(
            status_code=error.status,
            content={"error": {"code": error.code}},
            headers={"WWW-Authenticate": "Bearer"} if error.status == 401 else None,
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "stage": "M1-local" if settings.model_mode == "deepseek" else "M0",
            "model_mode": settings.model_mode,
            "memory_status": "recent_history_only"
            if settings.model_mode == "deepseek"
            else "not_implemented",
            "output_guard": "full_response"
            if settings.model_mode == "deepseek"
            else "fixture_whitelist",
        }

    @app.get("/ready")
    async def ready():
        try:
            async with conversation.db.sessions() as session:
                await session.execute(text("SELECT 1 FROM alembic_version LIMIT 1"))
        except Exception:
            return JSONResponse(status_code=503, content={"status": "database_unavailable"})
        return {"status": "ready", "model_mode": settings.model_mode}

    @app.get("/api/identity")
    async def identity(owner_id: Annotated[UUID, Depends(require_owner)]):
        return app.state.identity

    @app.post("/api/conversations", status_code=201)
    async def create_chat(body: CreateChat, owner_id: Annotated[UUID, Depends(require_owner)]):
        return await conversation.create_chat(owner_id, body.character_id, body.request_id)

    @app.get("/api/conversations")
    async def list_chats(owner_id: Annotated[UUID, Depends(require_owner)]):
        return await conversation.chats(owner_id)

    @app.post("/api/conversations/{chat_id}/turns", status_code=202)
    async def accept_turn(
        chat_id: UUID,
        body: CreateTurn,
        owner_id: Annotated[UUID, Depends(require_owner)],
    ):
        return await conversation.accept(
            owner_id, chat_id, body.request_id, body.content, model_mode=settings.model_mode
        )

    @app.get("/api/conversations/{chat_id}/messages")
    async def messages(chat_id: UUID, owner_id: Annotated[UUID, Depends(require_owner)]):
        return await conversation.messages(owner_id, chat_id)

    @app.get("/api/turns/{turn_id}")
    async def turn(turn_id: UUID, owner_id: Annotated[UUID, Depends(require_owner)]):
        return await conversation.get_turn(owner_id, turn_id)

    @app.post("/api/turns/{turn_id}/cancel")
    async def cancel(
        turn_id: UUID,
        body: WriteBody,
        owner_id: Annotated[UUID, Depends(require_owner)],
    ):
        return await conversation.cancel(owner_id, turn_id, body.request_id)

    @app.get("/api/turns/{turn_id}/events")
    async def events(
        turn_id: UUID,
        request: Request,
        owner_id: Annotated[UUID, Depends(require_owner)],
        after: Annotated[int | None, Query(ge=0)] = None,
        last_event_id: Annotated[str | None, Header()] = None,
    ):
        try:
            header_cursor = int(last_event_id) if last_event_id is not None else None
        except ValueError:
            raise DomainError("invalid_cursor", 422) from None
        if after is not None and header_cursor is not None and after != header_cursor:
            raise DomainError("invalid_cursor", 422)
        cursor = after if after is not None else (header_cursor or 0)
        initial = await conversation.events(owner_id, turn_id, cursor)

        async def stream():
            nonlocal cursor
            batch = initial
            last_heartbeat = time.monotonic()
            while not await request.is_disconnected():
                for event in batch:
                    yield (
                        f"id: {event.id}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"
                    )
                    cursor = event.id
                current = await conversation.get_turn(owner_id, turn_id)
                if current.status in TERMINAL and cursor >= current.next_event_id - 1:
                    return
                if time.monotonic() - last_heartbeat >= 5:
                    yield ": keep-alive\n\n"
                    last_heartbeat = time.monotonic()
                await asyncio.sleep(0.05)
                batch = await conversation.events(owner_id, turn_id, cursor)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app
