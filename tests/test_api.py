from fastapi.testclient import TestClient

from seo_blog_assistant.api import create_app
from seo_blog_assistant.providers.search import ResearchProviderError
from seo_blog_assistant.providers.text_model import TextModelError
from seo_blog_assistant.schemas import ArticleResult, ImageGenerationResult


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
