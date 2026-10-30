"""Shared workflow construction and article-generation application service."""

from typing import Protocol

from langchain_groq import ChatGroq
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from seo_blog_assistant.config import Settings, get_settings
from seo_blog_assistant.providers.image import (
    ImageProvider,
    ImageProviderError,
    create_image_provider,
)
from seo_blog_assistant.providers.search import (
    ResearchProvider,
    create_research_provider,
)
from seo_blog_assistant.providers.text_model import (
    LangChainStructuredTextModel,
    StructuredTextModel,
)
from seo_blog_assistant.schemas import ArticleResult, BlogRequest
from seo_blog_assistant.workflow.graph import build_blog_workflow_graph
from seo_blog_assistant.workflow.state import BlogWorkflowState

GROQ_MAX_COMPLETION_TOKENS = 2048


class BlogWorkflow(Protocol):
    """Runnable asynchronous workflow contract shared by API and UI."""

    async def ainvoke(
        self,
        input: BlogWorkflowState,
    ) -> BlogWorkflowState: ...


class WorkflowConfigurationError(RuntimeError):
    """Raised when required runtime provider configuration is missing."""


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
        if runtime_settings.text_model_api_key is None:
            raise WorkflowConfigurationError(
                "Set SEO_BLOG_GROQ_API_KEY or SEO_BLOG_TEXT_MODEL_API_KEY "
                "to configure text generation."
            )
        text_model = _create_groq_text_model(
            runtime_settings.text_model_name,
            runtime_settings.text_model_api_key,
            runtime_settings.text_model_timeout_seconds,
        )

    if search_provider is None and runtime_settings.search_provider is not None:
        search_provider = create_research_provider(runtime_settings)
    if image_provider is None and runtime_settings.image_provider is not None:
        try:
            image_provider = create_image_provider(runtime_settings)
        except ImageProviderError as exc:
            raise WorkflowConfigurationError(str(exc)) from None

    return build_blog_workflow_graph(
        text_model,
        search_provider=search_provider,
        image_provider=image_provider,
    )


def _create_groq_text_model(
    model_name: str,
    api_key: SecretStr,
    timeout_seconds: int,
) -> StructuredTextModel:
    model = ChatGroq(
        model=model_name,
        api_key=api_key.get_secret_value(),
        timeout=timeout_seconds,
        max_tokens=GROQ_MAX_COMPLETION_TOKENS,
    )
    return LangChainStructuredTextModel(model)


async def generate_article(
    request: BlogRequest,
    workflow: BlogWorkflow,
) -> ArticleResult:
    """Run the shared workflow and return its validated article result."""
    result = await workflow.ainvoke({"request": request})
    article = result.get("article")
    if article is None:
        raise RuntimeError("Blog workflow completed without producing an article.")
    return ArticleResult.model_validate(article)
