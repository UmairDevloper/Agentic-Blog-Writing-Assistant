import httpx
import pytest
from groq import APIConnectionError, APITimeoutError
from langchain_core.exceptions import OutputParserException
from pydantic import Field, create_model
from fastapi.testclient import TestClient

from seo_blog_assistant.api import (
    InMemoryArticleJobStore,
    JobStoreFullError,
    create_app,
)
from seo_blog_assistant.providers.search import ResearchProviderError
from seo_blog_assistant.providers.text_model import TextModelError
from seo_blog_assistant.schemas import (
    ArticleResult,
    BlogRequest,
    ImageGenerationResult,
    ResearchSource,
)


def _article() -> ArticleResult:
    return ArticleResult(
        title="A practical topic guide",
        meta_description="A useful overview with practical takeaways.",
        search_intent="Informational",
        primary_keyword="topic guide",
        related_keyword_ideas=["topic basics"],
        markdown="# A practical topic guide\n\n## Introduction\n\nA useful guide.\n",
        sources=[],
        image_result=ImageGenerationResult(status="skipped"),
    )


class FakeWorkflow:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, input: dict[str, object]) -> dict[str, object]:
        self.calls.append(input)
        return {"article": _article()}


def test_api_returns_validated_article_from_shared_workflow() -> None:
    workflow = FakeWorkflow()
    client = TestClient(create_app(workflow))

    response = client.post(
        "/articles",
        json={
            "topic": "A useful topic",
            "require_citations": True,
            "target_words": 900,
        },
    )

    assert response.status_code == 200
    assert response.json()["title"] == "A practical topic guide"
    assert response.json()["image_result"]["status"] == "skipped"
    assert workflow.calls[0]["request"].topic == "A useful topic"


def test_api_revalidates_article_instances_from_a_reloaded_schema_class() -> None:
    field_definitions = {}
    for name, model_field in ArticleResult.model_fields.items():
        if model_field.default_factory is not None:
            default = Field(default_factory=model_field.default_factory)
        elif model_field.is_required():
            default = ...
        else:
            default = model_field.default
        field_definitions[name] = (model_field.annotation, default)

    reloaded_article_class = create_model(
        "ArticleResult",
        __config__=ArticleResult.model_config,
        **field_definitions,
    )
    reloaded_article_class.model_rebuild(
        _types_namespace={"ResearchSource": ResearchSource}
    )
    article_from_reloaded_module = reloaded_article_class.model_validate(
        _article().model_dump(mode="python")
    )

    class ReloadedWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            return {"article": article_from_reloaded_module}

    response = TestClient(create_app(ReloadedWorkflow())).post(
        "/articles",
        json={"topic": "A useful topic"},
    )

    assert response.status_code == 200
    assert response.json()["title"] == "A practical topic guide"


def test_api_accepts_category_and_returns_category_metadata() -> None:
    class SportsWorkflow:
        async def ainvoke(self, input: dict[str, object]) -> dict[str, object]:
            request = input["request"]
            assert isinstance(request, BlogRequest)
            assert request.category == "sports"
            article = _article().model_copy(
                update={
                    "topic_category": "sports",
                    "content_angle": "season preview",
                }
            )
            return {"article": article}

    client = TestClient(create_app(SportsWorkflow()))
    response = client.post(
        "/articles",
        json={"topic": "Harbor City football", "category": "sports"},
    )

    assert response.status_code == 200
    assert response.json()["topic_category"] == "sports"
    assert response.json()["content_angle"] == "season preview"


def test_api_rejects_invalid_request_without_running_workflow() -> None:
    workflow = FakeWorkflow()
    client = TestClient(create_app(workflow))

    response = client.post("/articles", json={"topic": "x"})

    assert response.status_code == 422
    assert workflow.calls == []


def test_api_surfaces_research_provider_failure_without_success_response() -> None:
    class FailedWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            raise ResearchProviderError("Configure Tavily to enable research.")

    client = TestClient(create_app(FailedWorkflow()))

    response = client.post("/articles", json={"topic": "A current topic"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Configure Tavily to enable research."


def test_api_returns_rate_limit_status_for_groq_tpm_errors() -> None:
    class RateLimitedWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            raise TextModelError(
                "Groq token or rate limit exceeded.",
                status_code=429,
            )

    client = TestClient(create_app(RateLimitedWorkflow()))

    response = client.post("/articles", json={"topic": "A useful topic"})

    assert response.status_code == 429
    assert "Groq token or rate limit exceeded" in response.json()["detail"]


def test_api_returns_gateway_error_for_invalid_groq_structured_output() -> None:
    class InvalidOutputWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            raise TextModelError(
                "Groq returned an invalid structured response.",
                status_code=502,
            )

    client = TestClient(create_app(InvalidOutputWorkflow()))

    response = client.post("/articles", json={"topic": "A useful topic"})

    assert response.status_code == 502
    assert "invalid structured response" in response.json()["detail"]


def test_api_reports_missing_text_provider_configuration(
    monkeypatch,
) -> None:
    from seo_blog_assistant import api
    from seo_blog_assistant.application import WorkflowConfigurationError

    def fail_configuration():
        raise WorkflowConfigurationError(
            "Set SEO_BLOG_TEXT_MODEL_PROVIDER=openai to configure text generation."
        )

    monkeypatch.setattr(api, "build_workflow", fail_configuration)
    client = TestClient(create_app())

    response = client.post("/articles", json={"topic": "A useful topic"})

    assert response.status_code == 503
    assert "SEO_BLOG_TEXT_MODEL_PROVIDER" in response.json()["detail"]


@pytest.mark.parametrize(
    ("failure", "expected_status", "retry_after"),
    [
        (ValueError("Invalid generated structure"), 502, None),
        (
            APITimeoutError(
                httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            ),
            504,
            None,
        ),
        (
            APIConnectionError(
                message="Connection failed",
                request=httpx.Request(
                    "POST", "https://api.groq.com/openai/v1/chat/completions"
                ),
            ),
            502,
            None,
        ),
        (OutputParserException("Invalid model output"), 502, None),
        (
            TextModelError(
                "Groq rate limit hit.",
                status_code=429,
                retry_after_seconds=30,
            ),
            429,
            "30",
        ),
        (RuntimeError("Unexpected internal detail"), 500, None),
    ],
)
def test_api_returns_json_details_for_generation_failures(
    failure: Exception,
    expected_status: int,
    retry_after: str | None,
) -> None:
    class FailedWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            raise failure

    response = TestClient(create_app(FailedWorkflow())).post(
        "/articles",
        json={"topic": "A useful topic"},
    )

    assert response.status_code == expected_status
    assert isinstance(response.json()["detail"], str)
    assert response.text != "Internal Server Error"
    assert response.headers.get("Retry-After") == retry_after


def test_article_job_returns_completed_result_from_fake_workflow() -> None:
    client = TestClient(create_app(FakeWorkflow()))

    submitted = client.post(
        "/jobs",
        json={"topic": "A useful topic"},
    )

    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]
    assert isinstance(job_id, str)

    status_response = None
    for _ in range(10):
        status_response = client.get(f"/jobs/{job_id}")
        if status_response.json()["status"] == "completed":
            break

    assert status_response is not None
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"
    assert status_response.json()["result"]["title"] == "A practical topic guide"
    assert status_response.json()["error"] is None


def test_article_job_reports_failure_without_raw_internal_error() -> None:
    class FailedWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            raise ValueError("Invalid generated structure")

    client = TestClient(create_app(FailedWorkflow()))
    submitted = client.post("/jobs", json={"topic": "A useful topic"})

    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]
    status_response = None
    for _ in range(10):
        status_response = client.get(f"/jobs/{job_id}")
        if status_response.json()["status"] == "failed":
            break

    assert status_response is not None
    assert status_response.json()["status"] == "failed"
    assert "failed validation" in status_response.json()["error"]


def test_article_job_returns_not_found_for_unknown_id() -> None:
    response = TestClient(create_app(FakeWorkflow())).get("/jobs/not-a-job")

    assert response.status_code == 404
    assert response.json()["detail"]


def test_in_memory_job_store_enforces_capacity_and_ttl() -> None:
    current_time = 0.0
    store = InMemoryArticleJobStore(
        max_jobs=1,
        ttl_seconds=3600,
        clock=lambda: current_time,
    )

    first_job = store.create()
    with pytest.raises(JobStoreFullError):
        store.create()

    current_time = 3601.0
    assert store.get(first_job.job_id) is None
    assert store.create().job_id != first_job.job_id
