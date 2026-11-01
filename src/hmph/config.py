from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HMPH_",
        env_file=(".env", ".env.local"),
        extra="ignore",
        hide_input_in_errors=True,
    )

    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://hmph:hmph_local_only@127.0.0.1:54329/hmph"
    )
    api_token: SecretStr
    request_digest_key: SecretStr
    database_schema: str = Field(default="public", pattern=r"^[a-z][a-z0-9_]*$")
    model_mode: Literal["fake", "deepseek"] = "fake"
    source_timezone: str = Field(default="UTC", max_length=80)
    deepseek_api_key: SecretStr | None = None
    deepseek_model: Literal["deepseek-flash", "deepseek-v4-pro"] = "deepseek-flash"
    deepseek_max_tokens: int = Field(default=256, ge=1, le=512)
    fake_delay_seconds: float = Field(default=0.01, gt=0, le=10)
    lease_seconds: float = Field(default=15, gt=0, le=60)
    turn_timeout_seconds: float = Field(default=45, gt=0, le=300)

    @field_validator("source_timezone")
    @classmethod
    def valid_source_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("Invalid source timezone") from None
        return value

    @field_validator("api_token", "request_digest_key")
    @classmethod
    def token_must_be_long(cls, value: SecretStr):
        if len(value.get_secret_value()) < 32:
            raise ValueError("Credentials must be random strings of at least 32 characters")
        return value

    @model_validator(mode="after")
    def provider_must_be_configured(self):
        if self.model_mode == "deepseek" and (
            self.deepseek_api_key is None or not self.deepseek_api_key.get_secret_value().strip()
        ):
            raise ValueError("DeepSeek mode requires a local HMPH_DEEPSEEK_API_KEY")
        return self
