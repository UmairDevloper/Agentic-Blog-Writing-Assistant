"""Structured text-generation interface and LangChain adapter."""

from typing import Protocol, TypeVar

from groq import APIStatusError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class TextModelError(RuntimeError):
    """Sanitized text-provider error safe to report to an API caller."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class StructuredTextModel(Protocol):
    """Generates validated structured responses for workflow nodes."""

    async def generate_structured(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT: ...


class LangChainStructuredTextModel:
    """Adapts any LangChain chat model supporting structured output."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

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
        try:
            response = await structured_model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            if isinstance(response, schema):
                return response
            return schema.model_validate(response)
        except APIStatusError as exc:
            error_body = exc.body
            error_info = (
                error_body.get("error", {})
                if isinstance(error_body, dict)
                else {}
            )
            error_code = (
                error_info.get("code")
                if isinstance(error_info, dict)
                else None
            )
            if exc.status_code == 413 or error_code == "rate_limit_exceeded":
                raise TextModelError(
                    "Groq rate limit exceeded. Each article makes multiple text "
                    "generation requests. Wait for the provider quota to reset "
                    "before retrying; if it keeps happening, reduce the requested "
                    "article length or use a model/account with higher limits.",
                    status_code=429,
                ) from None
            if exc.status_code == 400:
                raise TextModelError(
                    "Groq rejected the structured-output request (HTTP 400). "
                    "Confirm the configured model supports Groq's strict JSON "
                    "Schema output.",
                    status_code=502,
                ) from None
            raise TextModelError(
                f"Groq text generation failed with HTTP {exc.status_code}.",
                status_code=502,
            ) from None
        except ValidationError:
            raise TextModelError(
                "Groq returned an invalid structured response. Retry the request "
                "or use a model with more reliable structured-output support.",
                status_code=502,
            ) from None
