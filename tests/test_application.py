import pytest
from pydantic import SecretStr

from seo_blog_assistant.application import (
    WorkflowConfigurationError,
    build_workflow,
)
from seo_blog_assistant.config import Settings
from seo_blog_assistant.providers.text_model import TextResult


class FakeStructuredTextModel:
    async def generate_structured(
        self,
        *,
        schema: type,
        system_prompt: str,
        user_prompt: str,
    ):
        raise AssertionError("The workflow factory must not call the text model.")

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> TextResult:
        raise AssertionError("The workflow factory must not call the text model.")


def test_build_workflow_requires_text_model_configuration() -> None:
    with pytest.raises(WorkflowConfigurationError, match="TEXT_MODEL_PROVIDER"):
        build_workflow(Settings(_env_file=None))


def test_build_workflow_can_inject_text_provider_without_credentials() -> None:
    workflow = build_workflow(
        Settings(_env_file=None),
        text_model=FakeStructuredTextModel(),
    )

    assert workflow is not None


def test_build_workflow_constructs_configured_langchain_groq_adapter() -> None:
    workflow = build_workflow(
        Settings(
            _env_file=None,
            text_model_provider="groq",
            text_model_name="test-model",
            text_model_api_key=SecretStr("test-api-key"),
        )
    )

    assert workflow is not None


def test_groq_model_limits_completion_tokens_for_free_tier_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seo_blog_assistant import application

    captured: dict[str, object] = {}

    class FakeChatGroq:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(application, "ChatGroq", FakeChatGroq)

    workflow = build_workflow(
        Settings(
            _env_file=None,
            text_model_provider="groq",
            text_model_name="test-model",
            text_model_api_key=SecretStr("test-api-key"),
        )
    )

    assert workflow is not None
    assert captured["max_tokens"] == 4096
    assert captured["max_retries"] == 0
    assert "reasoning_effort" not in captured


def test_groq_model_uses_reasoning_effort_for_gpt_oss_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seo_blog_assistant import application

    captured: dict[str, object] = {}

    class FakeChatGroq:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(application, "ChatGroq", FakeChatGroq)

    build_workflow(
        Settings(
            _env_file=None,
            text_model_provider="groq",
            text_model_name="openai/gpt-oss-20b",
            text_model_api_key=SecretStr("test-api-key"),
            text_model_reasoning_effort="medium",
        )
    )

    assert captured["reasoning_effort"] == "medium"


def test_groq_model_uses_optional_request_rate_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seo_blog_assistant import application

    captured: dict[str, object] = {}

    class FakeChatGroq:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(application, "ChatGroq", FakeChatGroq)

    workflow = build_workflow(
        Settings(
            _env_file=None,
            text_model_provider="groq",
            text_model_name="test-model",
            text_model_api_key=SecretStr("test-api-key"),
            text_model_requests_per_second=0.5,
            text_model_max_attempts=3,
            text_model_max_concurrency=1,
            text_model_max_completion_tokens=1024,
        )
    )

    assert workflow is not None
    assert isinstance(captured["rate_limiter"], application.InMemoryRateLimiter)
    assert captured["max_tokens"] == 1024
    assert captured["max_retries"] == 0


def test_non_strict_groq_model_compatibility_warning_is_logged_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from seo_blog_assistant import application

    monkeypatch.setattr(application, "_WARNED_NON_STRICT_MODELS", set())
    settings = Settings(
        _env_file=None,
        text_model_provider="groq",
        text_model_name="other/model",
        text_model_api_key=SecretStr("test-api-key"),
    )

    with caplog.at_level("WARNING", logger="seo_blog_assistant.application"):
        build_workflow(settings)
        build_workflow(settings)

    matching = [
        record
        for record in caplog.records
        if "does not support strict JSON Schema decoding" in record.message
    ]
    assert len(matching) == 1


def test_invalid_optional_image_configuration_does_not_block_workflow(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(
        _env_file=None,
        image_provider="huggingface",
        image_mode="generate",
    )

    with caplog.at_level("WARNING", logger="seo_blog_assistant.application"):
        workflow = build_workflow(
            settings,
            text_model=FakeStructuredTextModel(),
        )

    assert workflow is not None
    assert "Optional image provider is unavailable" in caplog.text


def test_placeholder_mode_never_constructs_image_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seo_blog_assistant import application

    def fail_if_created(_settings: Settings) -> None:
        raise AssertionError("placeholder mode must not construct an image provider")

    monkeypatch.setattr(application, "create_image_provider", fail_if_created)

    workflow = build_workflow(
        Settings(
            _env_file=None,
            image_mode="placeholder",
            image_provider="huggingface",
        ),
        text_model=FakeStructuredTextModel(),
    )

    assert workflow is not None
