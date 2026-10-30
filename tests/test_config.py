import pytest
from pydantic import SecretStr, ValidationError

from seo_blog_assistant.config import Settings


def test_settings_load_environment_values_without_exposing_secrets(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SEO_BLOG_APP_ENV", "test")
    monkeypatch.setenv("SEO_BLOG_TEXT_MODEL_API_KEY", "test-secret")
    monkeypatch.setenv("SEO_BLOG_HUGGINGFACE_TOKEN", "hf-test-secret")
    monkeypatch.setenv("SEO_BLOG_IMAGE_PROVIDER", "huggingface")

    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.text_model_api_key == SecretStr("test-secret")
    assert settings.huggingface_token == SecretStr("hf-test-secret")
    assert "test-secret" not in repr(settings)
    assert "hf-test-secret" not in repr(settings)


def test_settings_can_disable_project_env_loading() -> None:
    settings = Settings(_env_file=None, app_env="test")

    assert settings.app_env == "test"
    assert settings.huggingface_token is None


def test_settings_accept_hugging_face_image_provider() -> None:
    settings = Settings(
        _env_file=None,
        image_provider="huggingface",
        image_model_name="owner/open-image-model",
        huggingface_token="hf-placeholder",
    )

    assert settings.image_provider == "huggingface"
    assert settings.image_model_name == "owner/open-image-model"
    assert settings.huggingface_token == SecretStr("hf-placeholder")


def test_settings_accept_tavily_search_provider_and_bounds() -> None:
    settings = Settings(
        _env_file=None,
        search_provider="TAVILY",
        search_api_key=SecretStr("tvly-placeholder"),
        search_timeout_seconds=45,
        search_max_results=10,
    )

    assert settings.search_provider == "tavily"
    assert settings.search_timeout_seconds == 45
    assert settings.search_max_results == 10


def test_settings_accept_groq_text_model_configuration() -> None:
    settings = Settings(
        _env_file=None,
        text_model_provider="GROQ",
        text_model_name="test-model",
        text_model_api_key=SecretStr("test-api-key"),
        text_model_timeout_seconds=45,
    )

    assert settings.text_model_provider == "groq"
    assert settings.text_model_name == "test-model"
    assert settings.text_model_timeout_seconds == 45
    assert "test-api-key" not in repr(settings)


@pytest.mark.parametrize(
    "environment_variable",
    [
        "SEO_BLOG_GROQ_API_KEY",
        "GROQ_API_KEY",
        "SEO_BLOG_TEXT_MODEL_API_KEY",
    ],
)
def test_settings_load_groq_key_aliases(
    monkeypatch,
    environment_variable: str,
) -> None:
    monkeypatch.setenv(environment_variable, "groq-test-secret")

    settings = Settings(_env_file=None)

    assert settings.text_model_api_key == SecretStr("groq-test-secret")
    assert "groq-test-secret" not in repr(settings)


def test_settings_reject_out_of_range_search_limits() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, search_timeout_seconds=121)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, search_max_results=21)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, text_model_timeout_seconds=601)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, text_model_provider="openai")


def test_settings_reject_unsupported_image_provider() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, image_provider="local")


def test_settings_reject_invalid_log_level() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="VERBOSE")


def test_settings_load_project_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SEO_BLOG_LOG_LEVEL=DEBUG\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.log_level == "DEBUG"
