"""Shared workflow construction and article-generation application service."""

import asyncio
import logging
import time
import threading
from typing import Protocol

import httpx
from groq import APIConnectionError, APITimeoutError
from langchain_core.exceptions import OutputParserException
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ValidationError

from seo_blog_assistant.config import Settings, get_settings
from seo_blog_assistant.providers.image import (
    ImageProvider,
    ImageProviderError,
    create_image_provider,
)
from seo_blog_assistant.providers.search import (
    ResearchProvider,
    ResearchProviderError,
    create_research_provider,
)
from seo_blog_assistant.providers.text_model import (
    LangChainStructuredTextModel,
    StructuredTextModel,
    TextModelError,
)
from seo_blog_assistant.schemas import ArticleResult, BlogRequest
from seo_blog_assistant.workflow.graph import build_blog_workflow_graph
from seo_blog_assistant.workflow.state import BlogWorkflowState

logger = logging.getLogger(__name__)
_STRICT_JSON_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
_WARNED_NON_STRICT_MODELS: set[str] = set()
_MODEL_WARNING_LOCK = threading.Lock()


class BlogWorkflow(Protocol):
    """Runnable asynchronous workflow contract shared by API and UI."""

    async def ainvoke(
        self,
        input: BlogWorkflowState,
    ) -> BlogWorkflowState: ...


class WorkflowConfigurationError(RuntimeError):
    """Raised when required runtime provider configuration is missing."""


class ArticleGenerationError(RuntimeError):
    """Sanitized failure while running or validating an article workflow."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def build_workflow(
    settings: Settings | None = None,
    *,
    text_model: StructuredTextModel | None = None,
    search_provider: ResearchProvider | None = None,
    image_provider: ImageProvider | None = None,
) -> CompiledStateGraph[
    BlogWorkflowState,
    None,
    BlogWorkflowState,
    BlogWorkflowState,
]:
    """Build the configured workflow while allowing deterministic provider fakes."""
    runtime_settings = settings or get_settings()

    if text_model is None:
        if runtime_settings.text_model_provider != "groq":
            raise WorkflowConfigurationError(
                "Set SEO_BLOG_TEXT_MODEL_PROVIDER=groq to configure text generation."
            )
        if not runtime_settings.text_model_name:
            raise WorkflowConfigurationError(
                "Set SEO_BLOG_TEXT_MODEL_NAME to configure text generation."
            )
        _warn_if_model_lacks_strict_json(runtime_settings.text_model_name)
        if runtime_settings.text_model_api_key is None:
            raise WorkflowConfigurationError(
                "Set SEO_BLOG_GROQ_API_KEY or SEO_BLOG_TEXT_MODEL_API_KEY "
                "to configure text generation."
            )
        text_model = _create_groq_text_model(
            runtime_settings
        )

    if search_provider is None and runtime_settings.search_provider is not None:
        search_provider = create_research_provider(runtime_settings)
    if (
        runtime_settings.image_mode == "generate"
        and image_provider is None
        and runtime_settings.image_provider is not None
    ):
        try:
            image_provider = create_image_provider(runtime_settings)
        except ImageProviderError as exc:
            logger.warning(
                "Optional image provider is unavailable error_type=%s",
                type(exc).__name__,
            )
            image_provider = None

    return build_blog_workflow_graph(
        text_model,
        search_provider=search_provider,
        image_provider=image_provider,
        writing_mode=runtime_settings.writing_mode,
        max_words_per_call=runtime_settings.max_words_per_call,
        max_images=runtime_settings.max_images,
        settings=runtime_settings,
    )


def _warn_if_model_lacks_strict_json(model_name: str) -> None:
    if model_name in _STRICT_JSON_MODELS:
        return
    with _MODEL_WARNING_LOCK:
        if model_name in _WARNED_NON_STRICT_MODELS:
            return
        logger.warning(
            "Groq model %s does not support strict JSON Schema decoding; "
            "structured-output retries and repairs may be needed.",
            model_name,
        )
        _WARNED_NON_STRICT_MODELS.add(model_name)


def _create_groq_text_model(
    settings: Settings,
) -> StructuredTextModel:
    if settings.text_model_name is None or settings.text_model_api_key is None:
        raise WorkflowConfigurationError("Groq model settings are required.")

    model_options: dict[str, object] = {
        "model": settings.text_model_name,
        "api_key": settings.text_model_api_key.get_secret_value(),
        "timeout": settings.text_model_timeout_seconds,
        "max_retries": 0,
        "max_tokens": settings.text_model_max_completion_tokens,
    }
    if (
        settings.text_model_name.startswith("openai/gpt-oss")
        and settings.text_model_reasoning_effort is not None
    ):
        model_options["reasoning_effort"] = settings.text_model_reasoning_effort
    if settings.text_model_requests_per_second is not None:
        model_options["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=settings.text_model_requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    model = ChatGroq(**model_options)
    return LangChainStructuredTextModel(
        model,
        max_attempts=settings.text_model_max_attempts,
        max_concurrency=settings.text_model_max_concurrency,
    )


async def generate_article(
    request: BlogRequest,
    workflow: BlogWorkflow,
) -> ArticleResult:
    """Run the shared workflow and return its validated article result."""
    started_at = time.perf_counter()
    logger.info("Article workflow started topic_length=%s", len(request.topic))
    try:
        result = await workflow.ainvoke({"request": request})
        article = result.get("article")
        if article is None:
            raise RuntimeError("Blog workflow completed without producing an article.")
        validation_input = (
            article.model_dump(mode="python")
            if isinstance(article, BaseModel) and not isinstance(article, ArticleResult)
            else article
        )
        validated_article = ArticleResult.model_validate(validation_input)
    except (
        TextModelError,
        ResearchProviderError,
        WorkflowConfigurationError,
        ArticleGenerationError,
    ):
        raise
    except (ValueError, ValidationError, OutputParserException) as exc:
        message = str(exc)[:300]
        raise ArticleGenerationError(
            "The model produced content that failed validation: "
            f"{message}. Retry the request.",
            status_code=502,
        ) from None
    except (asyncio.TimeoutError, httpx.TimeoutException, APITimeoutError):
        raise ArticleGenerationError(
            "Article generation timed out. Retry the request.",
            status_code=504,
        ) from None
    except APIConnectionError:
        raise ArticleGenerationError(
            "Could not connect to the text-generation provider.",
            status_code=502,
        ) from None
    except Exception:
        logger.exception("Unexpected article workflow failure")
        raise ArticleGenerationError(
            "Unexpected error while generating the article. See server logs.",
            status_code=500,
        ) from None
    finally:
        logger.info(
            "Article workflow finished topic_length=%s duration_seconds=%.3f",
            len(request.topic),
            time.perf_counter() - started_at,
        )

    return validated_article
