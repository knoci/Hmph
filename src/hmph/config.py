from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HMPH_", env_file=".env", extra="ignore")

    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://hmph:hmph_local_only@127.0.0.1:54329/hmph"
    )
    api_token: SecretStr
    request_digest_key: SecretStr
    database_schema: str = Field(default="public", pattern=r"^[a-z][a-z0-9_]*$")
    model_mode: Literal["fake"] = "fake"
    fake_delay_seconds: float = Field(default=0.01, gt=0, le=10)
    lease_seconds: float = Field(default=15, gt=0, le=60)
    turn_timeout_seconds: float = Field(default=45, gt=0, le=300)

    @field_validator("api_token", "request_digest_key")
    @classmethod
    def token_must_be_long(cls, value: SecretStr):
        if len(value.get_secret_value()) < 32:
            raise ValueError("Credentials must be random strings of at least 32 characters")
        return value
