"""Explicit expression controls; no free-form prompt or inferred profile."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hmph.source_policy import SECRET


class ExpressionPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    nickname: str | None = Field(default=None, min_length=1, max_length=40)
    reply_length: Literal["short", "normal", "long"] = "normal"
    playfulness: Literal["low", "normal"] = "normal"
    support_mode: Literal["listen", "ask_first", "suggest"] = "ask_first"

    @field_validator("nickname")
    @classmethod
    def plain_nickname(cls, value):
        if value is not None and (
            not value.strip() or any(ord(c) < 32 for c in value) or SECRET.search(value)
        ):
            raise ValueError("Nickname must be plain non-secret text")
        return value


def expression_preferences(raw):
    return ExpressionPreferences.model_validate(
        {key: value for key, value in raw.items() if key in ExpressionPreferences.model_fields}
    )
