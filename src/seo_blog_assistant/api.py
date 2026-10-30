"""FastAPI entry point for the shared blog-generation workflow."""

from fastapi import FastAPI, HTTPException, Request, status

from seo_blog_assistant.application import (
    BlogWorkflow,
    WorkflowConfigurationError,
    build_workflow,
    generate_article,
)
from seo_blog_assistant.providers.search import ResearchProviderError
from seo_blog_assistant.providers.text_model import TextModelError
from seo_blog_assistant.schemas import ArticleResult, BlogRequest


def create_app(workflow: BlogWorkflow | None = None) -> FastAPI:
    """Create the API app, optionally injecting a workflow for tests or hosting."""
    app = FastAPI(title="SEO Blog Writing Assistant")
    app.state.workflow = workflow

    @app.post(
        "/articles",
        response_model=ArticleResult,
        status_code=status.HTTP_200_OK,
    )
    async def create_article(
        blog_request: BlogRequest,
        http_request: Request,
    ) -> ArticleResult:
        active_workflow: BlogWorkflow | None = http_request.app.state.workflow
        if active_workflow is None:
            try:
                active_workflow = build_workflow()
            except (WorkflowConfigurationError, ResearchProviderError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from None
            http_request.app.state.workflow = active_workflow

        try:
            return await generate_article(blog_request, active_workflow)
        except ResearchProviderError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from None
        except TextModelError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=str(exc),
            ) from None

    return app


app = create_app()
