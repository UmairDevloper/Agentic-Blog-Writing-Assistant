import pytest
from pydantic import SecretStr

from seo_blog_assistant.application import (
    WorkflowConfigurationError,
    build_workflow,
)
from seo_blog_assistant.config import Settings


class FakeStructuredTextModel:
    async def generate_structured(
        self,
        *,
        schema: type,
        system_prompt: str,
        user_prompt: str,
    ):
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
    assert captured["max_tokens"] == application.GROQ_MAX_COMPLETION_TOKENS
