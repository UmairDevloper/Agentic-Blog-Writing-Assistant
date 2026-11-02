import asyncio

import httpx
import pytest
from groq import APIStatusError
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError

from seo_blog_assistant.providers.text_model import (
    LangChainStructuredTextModel,
    TextModelError,
    TextResult,
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

    with pytest.raises(TextModelError, match="per-minute token limit") as error:
        asyncio.run(
            model.generate_structured(
                schema=ExampleResponse,
                system_prompt="System prompt",
                user_prompt="User prompt",
            )
        )

    assert error.value.status_code == 429
    assert "organization details and quota contents" in str(error.value)
    assert "sensitive provider details" not in str(error.value)
    assert groq_model.options == {"method": "json_schema", "strict": True}


def test_groq_bad_request_identifies_structured_output_support() -> None:
    model = LangChainStructuredTextModel(BadRequestGroqModel())

    with pytest.raises(TextModelError, match="could not be parsed or validated") as error:
        asyncio.run(
            model.generate_structured(
                schema=ExampleResponse,
                system_prompt="System prompt",
                user_prompt="User prompt",
            )
        )

    assert error.value.status_code == 502
    assert "provider-specific details" in str(error.value)


def test_invalid_structured_output_is_sanitized_for_callers() -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    model = LangChainStructuredTextModel(
        InvalidOutputGroqModel(),
        sleep=no_sleep,
    )

    with pytest.raises(TextModelError, match="could not be parsed or validated") as error:
        asyncio.run(
            model.generate_structured(
                schema=ExampleResponse,
                system_prompt="System prompt",
                user_prompt="User prompt",
            )
        )

    assert error.value.status_code == 502


def _status_error(status_code: int, *, retry_after: str | None = None) -> APIStatusError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        status_code,
        headers=headers,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return APIStatusError(
        "provider failure",
        response=response,
        body={"error": {"code": "rate_limit_exceeded", "message": "limited"}},
    )


def test_text_model_retries_429_using_retry_after_without_real_sleep() -> None:
    class RateLimitThenSuccess:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages: list[object]) -> object:
            self.calls += 1
            if self.calls == 1:
                raise _status_error(429, retry_after="2")
            return ExampleResponse(value="ok")

    runnable = RateLimitThenSuccess()

    class Model:
        def with_structured_output(
            self,
            _schema: type[BaseModel],
            **_kwargs: object,
        ) -> RateLimitThenSuccess:
            return runnable

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    model = LangChainStructuredTextModel(Model(), sleep=fake_sleep)
    result = asyncio.run(
        model.generate_structured(
            schema=ExampleResponse,
            system_prompt="System prompt",
            user_prompt="User prompt",
        )
    )

    assert result == ExampleResponse(value="ok")
    assert runnable.calls == 2
    assert sleep_calls == [2.0]


def test_text_model_does_not_retry_long_retry_after() -> None:
    class AlwaysRateLimited:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, _messages: list[object]) -> object:
            self.calls += 1
            raise _status_error(429, retry_after="3600")

    runnable = AlwaysRateLimited()

    class Model:
        def with_structured_output(
            self,
            _schema: type[BaseModel],
            **_kwargs: object,
        ) -> AlwaysRateLimited:
            return runnable

    async def unexpected_sleep(_delay: float) -> None:
        raise AssertionError("long provider reset must not be retried")

    model = LangChainStructuredTextModel(Model(), sleep=unexpected_sleep)
    with pytest.raises(TextModelError) as error:
        asyncio.run(
            model.generate_structured(
                schema=ExampleResponse,
                system_prompt="System prompt",
                user_prompt="User prompt",
            )
        )

    assert error.value.status_code == 429
    assert error.value.retry_after_seconds == 3600
    assert runnable.calls == 1


def test_text_model_limits_concurrent_calls() -> None:
    class ConcurrentRunnable:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def ainvoke(self, _messages: list[object]) -> object:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return ExampleResponse(value="ok")

    runnable = ConcurrentRunnable()

    class Model:
        def with_structured_output(
            self,
            _schema: type[BaseModel],
            **_kwargs: object,
        ) -> ConcurrentRunnable:
            return runnable

    adapter = LangChainStructuredTextModel(Model(), max_concurrency=2)

    async def run_calls() -> None:
        await asyncio.gather(
            *(
                adapter.generate_structured(
                    schema=ExampleResponse,
                    system_prompt="System prompt",
                    user_prompt=f"Request {index}",
                )
                for index in range(10)
            )
        )

    asyncio.run(run_calls())
    assert runnable.max_active == 2


def test_plain_text_generation_binds_tokens_and_strips_think_blocks() -> None:
    class Runnable:
        async def ainvoke(self, _messages: list[object]) -> AIMessage:
            return AIMessage(
                content=[
                    {"type": "text", "text": "Draft <think>private reasoning</think>"},
                    {"type": "text", "text": " Article"},
                ]
            )

    class Model:
        def __init__(self) -> None:
            self.max_tokens: int | None = None

        def bind(self, **kwargs: object) -> Runnable:
            self.max_tokens = int(kwargs["max_tokens"])
            return Runnable()

    groq_model = Model()
    adapter = LangChainStructuredTextModel(groq_model)  # type: ignore[arg-type]
    result = asyncio.run(
        adapter.generate_text(
            system_prompt="Write an article.",
            user_prompt='{"topic":"Example"}',
            max_tokens=1200,
        )
    )

    assert result == TextResult(text="Draft  Article", truncated=False)
    assert groq_model.max_tokens == 1200


def test_plain_text_generation_reports_provider_truncation() -> None:
    class Model:
        def bind(self, **_kwargs: object):
            class Runnable:
                async def ainvoke(self, _messages: list[object]) -> AIMessage:
                    return AIMessage(
                        content="Partial article",
                        response_metadata={"finish_reason": "length"},
                    )

            return Runnable()

    adapter = LangChainStructuredTextModel(Model())  # type: ignore[arg-type]
    result = asyncio.run(
        adapter.generate_text(
            system_prompt="Write an article.",
            user_prompt="Write this.",
            max_tokens=800,
        )
    )

    assert result.text == "Partial article"
    assert result.truncated is True


def test_plain_text_generation_retries_and_uses_shared_concurrency_limit() -> None:
    class Runnable:
        def __init__(self) -> None:
            self.calls = 0
            self.active = 0
            self.max_active = 0

        async def ainvoke(self, _messages: list[object]) -> AIMessage:
            self.calls += 1
            call_number = self.calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            if call_number == 1:
                raise _status_error(503)
            return AIMessage(content="Complete article")

    runnable = Runnable()

    class Model:
        def bind(self, **_kwargs: object) -> Runnable:
            return runnable

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    adapter = LangChainStructuredTextModel(
        Model(),  # type: ignore[arg-type]
        max_attempts=2,
        max_concurrency=2,
        sleep=fake_sleep,
    )

    async def invoke_many() -> list[TextResult]:
        return await asyncio.gather(
            *(
                adapter.generate_text(
                    system_prompt="Write.",
                    user_prompt=f"Article {index}",
                    max_tokens=800,
                )
                for index in range(5)
            )
        )

    results = asyncio.run(invoke_many())

    assert all(result.text == "Complete article" for result in results)
    assert runnable.calls == 6
    assert runnable.max_active == 2
    assert len(sleeps) == 1
