"""Structured text-generation interface and resilient LangChain adapter."""

import asyncio
import logging
import re
import threading
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from groq import APIConnectionError, APIStatusError, APITimeoutError
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from seo_blog_assistant.providers.retry import RetryDecision, retry_async

SchemaT = TypeVar("SchemaT", bound=BaseModel)
ResultT = TypeVar("ResultT")
logger = logging.getLogger(__name__)
_SECRET_PATTERN = re.compile(
    r"(?i)(bearer\s+|(?:gsk_|hf_|tvly-))[A-Za-z0-9._-]+"
)
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class TextResult:
    """Plain-text response and provider truncation signal."""

    text: str
    truncated: bool


class TextModelError(RuntimeError):
    """Sanitized text-provider error safe to report to an API caller."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class StructuredTextModel(Protocol):
    """Generates validated structured responses for workflow nodes."""

    async def generate_structured(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT: ...

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> TextResult: ...


class LangChainStructuredTextModel:
    """Adapts a LangChain chat model with retries and bounded concurrency."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        max_attempts: int = 4,
        max_concurrency: int = 2,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        self._model = model
        self._max_attempts = max_attempts
        self._max_concurrency = max_concurrency
        self._sleep = sleep
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()
        self._semaphore_lock = threading.Lock()

    async def generate_structured(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        structured_model = self._model.with_structured_output(
            schema,
            method="json_schema",
            strict=True,
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        def parse_response(response: object) -> SchemaT:
            if isinstance(response, schema):
                return response
            return schema.model_validate(response)

        result = await self._invoke_with_retries(
            operation=lambda: structured_model.ainvoke(messages),
            parse_response=parse_response,
            operation_name=schema.__name__,
        )
        return result

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> TextResult:
        """Generate plain Markdown text using the same retry/concurrency policy."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least one")
        model = self._model.bind(max_tokens=max_tokens)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return await self._invoke_with_retries(
            operation=lambda: model.ainvoke(messages),
            parse_response=_parse_text_response,
            operation_name="plain_text",
        )

    async def _invoke_with_retries(
        self,
        *,
        operation: Callable[[], Awaitable[object]],
        parse_response: Callable[[object], ResultT],
        operation_name: str,
    ) -> ResultT:
        attempts = 0
        json_validation_retries = 0

        async def invoke() -> ResultT:
            nonlocal attempts
            attempts += 1
            try:
                semaphore = self._semaphore_for_running_loop()
                async with semaphore:
                    response = await operation()
                result = parse_response(response)
            except Exception as exc:
                status_code = (
                    exc.status_code if isinstance(exc, APIStatusError) else None
                )
                logger.warning(
                    "Text provider attempt failed provider=groq operation=%s "
                    "attempt=%s status=%s error_type=%s",
                    operation_name,
                    attempts,
                    status_code,
                    type(exc).__name__,
                )
                raise
            logger.info(
                "Text provider attempt succeeded provider=groq operation=%s "
                "attempt=%s",
                operation_name,
                attempts,
            )
            return result

        def classify(exc: Exception) -> RetryDecision | None:
            nonlocal json_validation_retries
            if isinstance(
                exc,
                (
                    OutputParserException,
                    ValidationError,
                    APITimeoutError,
                    APIConnectionError,
                ),
            ):
                return RetryDecision()
            if not isinstance(exc, APIStatusError):
                return None

            if exc.status_code == 429:
                retry_after = _retry_after_seconds(exc)
                if retry_after is not None and retry_after > 60:
                    return None
                return RetryDecision(
                    delay_seconds=float(retry_after) if retry_after is not None else None
                )
            if exc.status_code in {500, 502, 503, 504}:
                return RetryDecision()
            if (
                exc.status_code == 400
                and _provider_error_code(exc) == "json_validate_failed"
                and json_validation_retries < 1
            ):
                json_validation_retries += 1
                return RetryDecision()
            return None

        try:
            return await retry_async(
                invoke,
                max_attempts=self._max_attempts,
                classify=classify,
                sleep=self._sleep,
                logger=logger,
                label=f"groq:{operation_name}",
            )
        except Exception as exc:
            raise _map_text_error(
                exc,
                operation_name=operation_name,
                attempts=attempts,
            ) from None

    def _semaphore_for_running_loop(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._semaphore_lock:
            semaphore = self._semaphores.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._max_concurrency)
                self._semaphores[loop] = semaphore
            return semaphore


def _parse_text_response(response: object) -> TextResult:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                part_text = part.get("text")
                if isinstance(part_text, str):
                    text_parts.append(part_text)
            else:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    text_parts.append(part_text)
        text = "".join(text_parts)
    else:
        raise ValueError("Text provider returned no text content.")

    metadata = getattr(response, "response_metadata", None)
    finish_reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
    return TextResult(
        text=_THINK_BLOCK_PATTERN.sub("", text).strip(),
        truncated=finish_reason == "length",
    )


def _provider_error_code(exc: APIStatusError) -> str | None:
    body = exc.body
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _retry_after_seconds(exc: APIStatusError) -> int | None:
    try:
        value = exc.response.headers.get("retry-after")
        return max(0, int(float(value))) if value is not None else None
    except (TypeError, ValueError):
        return None


def _provider_message(exc: APIStatusError) -> str | None:
    body = exc.body
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str) or not message:
        return None
    return _SECRET_PATTERN.sub("[redacted]", message[:300])


def _map_text_error(
    exc: Exception,
    *,
    operation_name: str,
    attempts: int,
) -> TextModelError:
    if isinstance(exc, APITimeoutError):
        return TextModelError(
            "Groq timed out while generating a response.",
            status_code=504,
        )
    if isinstance(exc, APIConnectionError):
        return TextModelError(
            "Could not connect to Groq while generating a response.",
            status_code=502,
        )
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        provider_message = _provider_message(exc)
        suffix = f" Provider message: {provider_message}" if provider_message else ""
        if status == 413:
            return TextModelError(
                "The request is larger than the model's per-minute token limit; "
                "waiting will not help. Reduce target_words or use a model with a "
                "higher limit."
                + suffix,
                status_code=429,
            )
        if status == 429 or _provider_error_code(exc) == "rate_limit_exceeded":
            retry_after = _retry_after_seconds(exc)
            retry_text = (
                f" Retry after {retry_after} seconds."
                if retry_after is not None
                else ""
            )
            return TextModelError(
                f"Groq rate limit hit after {attempts} attempts.{retry_text} "
                "Lower SEO_BLOG_TEXT_MODEL_MAX_CONCURRENCY, shorten the article, "
                "or use a model/account with a higher limit."
                + suffix,
                status_code=429,
                retry_after_seconds=retry_after,
            )
        if status in {401, 403}:
            return TextModelError(
                "Groq rejected access. Check the API key and model access." + suffix,
                status_code=502,
            )
        if status in {500, 502, 503, 504}:
            return TextModelError(
                f"Groq upstream error with HTTP status {status} after {attempts} "
                f"attempts.{suffix}",
                status_code=502,
            )
        if status == 400 or _is_parse_error(exc):
            return TextModelError(
                f"Model output for {operation_name} could not be parsed "
                f"or validated after {attempts} attempts.{suffix}",
                status_code=502,
            )
        return TextModelError(
            f"Groq text generation failed with HTTP {status}.{suffix}",
            status_code=502,
        )
    if isinstance(exc, (OutputParserException, ValidationError)):
        return TextModelError(
            f"Model output for {operation_name} could not be parsed or "
            f"validated after {attempts} attempts.",
            status_code=502,
        )
    return TextModelError(
        f"Groq text generation failed with {type(exc).__name__}.",
        status_code=502,
    )


def _is_parse_error(exc: APIStatusError) -> bool:
    return (
        exc.status_code == 400
        and _provider_error_code(exc) == "json_validate_failed"
    )
