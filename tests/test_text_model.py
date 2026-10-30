import asyncio

import httpx
import pytest
from groq import APIStatusError
from pydantic import BaseModel, ValidationError

from seo_blog_assistant.providers.text_model import (
    LangChainStructuredTextModel,
    TextModelError,
)


class ExampleResponse(BaseModel):
    value: str


class FailedRunnable:
    async def ainvoke(self, _messages: list[object]) -> object:
        response = httpx.Response(
            413,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )
        raise APIStatusError(
            "sensitive provider details should not escape",
            response=response,
            body={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "organization details and quota contents",
                }
            },
        )


class FailedGroqModel:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def with_structured_output(
        self,
        _schema: type[BaseModel],
        **kwargs: object,
    ) -> FailedRunnable:
        self.options = kwargs
        return FailedRunnable()


class InvalidOutputRunnable:
    async def ainvoke(self, _messages: list[object]) -> object:
        ExampleResponse.model_validate({"unexpected": "invalid"})


class InvalidOutputGroqModel:
    def with_structured_output(
        self,
        _schema: type[BaseModel],
        **_kwargs: object,
    ) -> InvalidOutputRunnable:
        return InvalidOutputRunnable()


class BadRequestRunnable:
    async def ainvoke(self, _messages: list[object]) -> object:
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )
        raise APIStatusError(
            "provider response must not be forwarded",
            response=response,
            body={"error": {"message": "provider-specific details"}},
        )


class BadRequestGroqModel:
    def with_structured_output(
        self,
        _schema: type[BaseModel],
        **_kwargs: object,
    ) -> BadRequestRunnable:
        return BadRequestRunnable()


def test_groq_rate_limit_error_is_sanitized_for_callers() -> None:
    groq_model = FailedGroqModel()
    model = LangChainStructuredTextModel(groq_model)

    with pytest.raises(TextModelError, match="rate limit") as error:
        asyncio.run(
            model.generate_structured(
                schema=ExampleResponse,
                system_prompt="System prompt",
                user_prompt="User prompt",
            )
        )

    assert error.value.status_code == 429
    assert "organization details" not in str(error.value)
    assert "sensitive provider details" not in str(error.value)
    assert groq_model.options == {"method": "json_schema", "strict": True}


def test_groq_bad_request_identifies_structured_output_support() -> None:
    model = LangChainStructuredTextModel(BadRequestGroqModel())

    with pytest.raises(TextModelError, match="strict JSON Schema output") as error:
        asyncio.run(
            model.generate_structured(
                schema=ExampleResponse,
                system_prompt="System prompt",
                user_prompt="User prompt",
            )
        )

    assert error.value.status_code == 502
    assert "provider-specific details" not in str(error.value)


def test_invalid_structured_output_is_sanitized_for_callers() -> None:
    model = LangChainStructuredTextModel(InvalidOutputGroqModel())

    with pytest.raises(TextModelError, match="invalid structured response") as error:
        asyncio.run(
            model.generate_structured(
                schema=ExampleResponse,
                system_prompt="System prompt",
                user_prompt="User prompt",
            )
        )

    assert error.value.status_code == 502
