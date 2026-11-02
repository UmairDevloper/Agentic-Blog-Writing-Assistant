"""FastAPI entry point for synchronous articles and in-memory jobs."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from seo_blog_assistant.application import (
    ArticleGenerationError,
    BlogWorkflow,
    WorkflowConfigurationError,
    build_workflow,
    generate_article,
)
from seo_blog_assistant.config import get_settings
from seo_blog_assistant.logging_config import configure_logging
from seo_blog_assistant.providers.search import ResearchProviderError
from seo_blog_assistant.providers.text_model import TextModelError
from seo_blog_assistant.schemas import ArticleResult, BlogRequest

logger = logging.getLogger(__name__)
JOB_TTL_SECONDS = 60 * 60
MAX_STORED_JOBS = 100
JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class ArticleJob:
    """Mutable in-memory state for one background article-generation job."""

    job_id: str
    status: JobStatus
    created_at: float
    result: ArticleResult | None = None
    error: str | None = None


class JobCreatedResponse(BaseModel):
    """Identifier returned when an article job is accepted."""

    job_id: str


class JobStatusResponse(BaseModel):
    """Current state and optional outcome of an article-generation job."""

    job_id: str
    status: JobStatus
    result: ArticleResult | None = None
    error: str | None = None


class JobStoreFullError(RuntimeError):
    """Raised when the in-memory job store has reached its configured cap."""


class InMemoryArticleJobStore:
    """Single-process job store with an expiry time and a strict size cap."""

    def __init__(
        self,
        *,
        max_jobs: int = MAX_STORED_JOBS,
        ttl_seconds: int = JOB_TTL_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self._max_jobs = max_jobs
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._jobs: dict[str, ArticleJob] = {}

    def create(self) -> ArticleJob:
        self._remove_expired()
        if len(self._jobs) >= self._max_jobs:
            raise JobStoreFullError(
                "The in-memory article job limit is full. Try again later."
            )
        job = ArticleJob(
            job_id=uuid4().hex,
            status="pending",
            created_at=self._clock(),
        )
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> ArticleJob | None:
        self._remove_expired()
        return self._jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.status = "running"

    def set_completed(self, job_id: str, result: ArticleResult) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.status = "completed"
            job.result = result

    def set_failed(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.status = "failed"
            job.error = error

    def _remove_expired(self) -> None:
        cutoff = self._clock() - self._ttl_seconds
        expired_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.created_at <= cutoff
        ]
        for job_id in expired_ids:
            del self._jobs[job_id]


def create_app(workflow: BlogWorkflow | None = None) -> FastAPI:
    """Create the API app, optionally injecting a workflow for tests or hosting."""
    configure_logging(get_settings().log_level)
    app = FastAPI(title="SEO Blog Writing Assistant")
    app.state.workflow = workflow
    app.state.job_store = InMemoryArticleJobStore()
    app.state.job_tasks = set()

    def get_active_workflow(http_request: Request) -> BlogWorkflow:
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
        return active_workflow

    @app.post(
        "/articles",
        response_model=ArticleResult,
        status_code=status.HTTP_200_OK,
    )
    async def create_article(
        blog_request: BlogRequest,
        http_request: Request,
    ) -> ArticleResult:
        active_workflow = get_active_workflow(http_request)
        try:
            return await generate_article(blog_request, active_workflow)
        except ResearchProviderError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from None
        except TextModelError as exc:
            raise _text_model_http_exception(exc) from None
        except ArticleGenerationError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.message,
            ) from None
        except WorkflowConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from None

    @app.post(
        "/jobs",
        response_model=JobCreatedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_article_job(
        blog_request: BlogRequest,
        http_request: Request,
    ) -> JobCreatedResponse:
        active_workflow = get_active_workflow(http_request)
        try:
            job = http_request.app.state.job_store.create()
        except JobStoreFullError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from None

        task = asyncio.create_task(
            _run_article_job(
                job.job_id,
                blog_request,
                active_workflow,
                http_request.app.state.job_store,
            )
        )
        http_request.app.state.job_tasks.add(task)
        task.add_done_callback(http_request.app.state.job_tasks.discard)
        return JobCreatedResponse(job_id=job.job_id)

    @app.get(
        "/jobs/{job_id}",
        response_model=JobStatusResponse,
    )
    async def get_article_job(
        job_id: str,
        http_request: Request,
    ) -> JobStatusResponse:
        job = http_request.app.state.job_store.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Article job was not found or has expired.",
            )
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            result=job.result,
            error=job.error,
        )

    return app


def _text_model_http_exception(exc: TextModelError) -> HTTPException:
    headers = (
        {"Retry-After": str(exc.retry_after_seconds)}
        if exc.retry_after_seconds is not None
        else None
    )
    logger.warning(
        "Article request failed status=%s failure_type=%s",
        exc.status_code,
        type(exc).__name__,
    )
    return HTTPException(
        status_code=exc.status_code,
        detail=str(exc),
        headers=headers,
    )


async def _run_article_job(
    job_id: str,
    request: BlogRequest,
    workflow: BlogWorkflow,
    job_store: InMemoryArticleJobStore,
) -> None:
    job_store.set_running(job_id)
    try:
        article = await generate_article(request, workflow)
    except (ResearchProviderError, TextModelError, ArticleGenerationError) as exc:
        job_store.set_failed(job_id, str(exc))
    except WorkflowConfigurationError as exc:
        job_store.set_failed(job_id, str(exc))
    except Exception:
        logger.exception("Unexpected background article job failure")
        job_store.set_failed(
            job_id,
            "Unexpected error while generating the article. See server logs.",
        )
    else:
        job_store.set_completed(job_id, article)


app = create_app()
