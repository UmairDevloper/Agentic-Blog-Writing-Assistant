import asyncio
import base64
from io import BytesIO

import pytest
import httpx
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from pydantic import SecretStr, ValidationError

from seo_blog_assistant.config import Settings
from seo_blog_assistant.providers.image import (
    HuggingFaceImageProvider,
    ImageProviderError,
    create_image_provider,
)
from seo_blog_assistant.schemas import GeneratedImage


def _png_image() -> Image.Image:
    return Image.new("RGB", (2, 2), color="blue")


class FakeInferenceClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None]] = []

    def text_to_image(
        self,
        prompt: str,
        *,
        model: str,
        num_inference_steps: int | None = None,
    ) -> Image.Image:
        self.calls.append((prompt, model, num_inference_steps))
        return _png_image()


def test_huggingface_provider_returns_validated_png_without_network() -> None:
    client = FakeInferenceClient()
    provider = HuggingFaceImageProvider(
        "owner/open-image-model",
        SecretStr("hf-test-secret"),
        inference_provider="fal-ai",
        num_inference_steps=7,
        timeout_seconds=30,
        client=client,
    )

    result = asyncio.run(
        provider.generate_image("Editorial illustration", "A blue square")
    )

    assert client.calls == [
        ("Editorial illustration", "owner/open-image-model", 7)
    ]
    assert result.model_name == "owner/open-image-model"
    assert result.media_type == "image/png"
    image_data = base64.b64decode(result.image_base64, validate=True)
    with Image.open(BytesIO(image_data)) as image:
        assert image.format == "PNG"
        assert image.size == (2, 2)


def test_huggingface_provider_configures_provider_and_passes_model_and_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seo_blog_assistant.providers import image as image_module

    captured: dict[str, object] = {}
    clients: list[FakeInferenceClient] = []

    class FakeConfiguredInferenceClient(FakeInferenceClient):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            captured.update(kwargs)
            clients.append(self)

    monkeypatch.setattr(image_module, "InferenceClient", FakeConfiguredInferenceClient)

    provider = create_image_provider(
        Settings(
            _env_file=None,
            image_provider="huggingface",
            image_model_name="owner/open-image-model",
            image_inference_provider="fal-ai",
            image_num_inference_steps=7,
            huggingface_token=SecretStr("hf-test-secret"),
            image_timeout_seconds=45,
        )
    )
    asyncio.run(provider.generate_image("Editorial scene", "A test image"))

    assert captured == {
        "provider": "fal-ai",
        "token": "hf-test-secret",
        "timeout": 45,
    }
    assert clients[0].calls == [
        ("Editorial scene", "owner/open-image-model", 7)
    ]


def test_huggingface_provider_sanitizes_hosted_inference_errors() -> None:
    class FailedInferenceClient:
        def text_to_image(
            self,
            _prompt: str,
            *,
            model: str,
            num_inference_steps: int | None = None,
        ) -> Image.Image:
            raise TimeoutError("request failed with hf-private-token")

    provider = HuggingFaceImageProvider(
        "owner/open-image-model",
        SecretStr("hf-private-token"),
        client=FailedInferenceClient(),
    )

    with pytest.raises(ImageProviderError, match="request timed out") as error:
        asyncio.run(provider.generate_image("An image", "An image"))

    assert "hf-private-token" not in str(error.value)


def test_huggingface_provider_sanitizes_unexpected_model_errors() -> None:
    class UnsupportedModelClient:
        def text_to_image(
            self,
            _prompt: str,
            *,
            model: str,
            num_inference_steps: int | None = None,
        ) -> Image.Image:
            raise ValueError("private account details")

    provider = HuggingFaceImageProvider(
        "owner/open-image-model",
        SecretStr("hf-test-secret"),
        client=UnsupportedModelClient(),
    )

    with pytest.raises(ImageProviderError, match="ValueError") as error:
        asyncio.run(provider.generate_image("An image", "An image"))

    assert "private account details" not in str(error.value)


def test_huggingface_provider_does_not_forward_exception_message() -> None:
    class FailedInferenceClient:
        def text_to_image(
            self,
            _prompt: str,
            *,
            model: str,
            num_inference_steps: int | None = None,
        ) -> Image.Image:
            raise ImageProviderError("hf-private-token and private provider body")

    provider = HuggingFaceImageProvider(
        "owner/open-image-model",
        SecretStr("hf-private-token"),
        client=FailedInferenceClient(),
    )

    with pytest.raises(ImageProviderError) as error:
        asyncio.run(provider.generate_image("private prompt", "An image"))

    assert "ImageProviderError" in str(error.value)
    assert "hf-private-token" not in str(error.value)
    assert "private provider body" not in str(error.value)
    assert "private prompt" not in str(error.value)


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "token rejected"),
        (403, "token rejected"),
        (402, "monthly credits exhausted"),
        (404, "model not available"),
        (422, "model not available"),
        (429, "rate limited"),
        (503, "provider error \\(503\\)"),
    ],
)
def test_huggingface_provider_maps_hosted_http_statuses(
    status_code: int,
    message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailedInferenceClient:
        def text_to_image(
            self,
            _prompt: str,
            *,
            model: str,
            num_inference_steps: int | None = None,
        ) -> Image.Image:
            response = httpx.Response(
                status_code,
                request=httpx.Request("POST", "https://api-inference.huggingface.co"),
            )
            raise HfHubHTTPError(
                "private provider response hf-test-secret prompt text",
                response=response,
            )

    provider = HuggingFaceImageProvider(
        "owner/open-image-model",
        SecretStr("hf-test-secret"),
        inference_provider="fal-ai",
        client=FailedInferenceClient(),
    )

    with pytest.raises(ImageProviderError, match=message) as error:
        asyncio.run(provider.generate_image("private prompt", "An image"))

    assert "private provider response" not in str(error.value)
    assert "hf-test-secret" not in str(error.value)
    assert "private prompt" not in str(error.value)
    assert "owner/open-image-model" in str(error.value)
    assert "fal-ai" in str(error.value)
    assert f"type=HfHubHTTPError status={status_code}" in caplog.text
    assert "hf-test-secret" not in caplog.text
    assert "private prompt" not in caplog.text


@pytest.mark.parametrize(
    ("image_data", "message"),
    [
        ("not-base64", "valid base64"),
        (base64.b64encode(b"not an image").decode("ascii"), "valid PNG"),
    ],
)
def test_generated_image_rejects_invalid_image_payload(
    image_data: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        GeneratedImage(
            model_name="owner/open-image-model",
            prompt="A useful illustration",
            alt_text="An illustration",
            image_base64=image_data,
        )


def test_image_provider_factory_requires_provider_model_and_token() -> None:
    with pytest.raises(ImageProviderError, match="IMAGE_PROVIDER=huggingface"):
        create_image_provider(Settings(_env_file=None))
    with pytest.raises(ImageProviderError, match="IMAGE_MODEL_NAME"):
        create_image_provider(Settings(_env_file=None, image_provider="huggingface"))
    with pytest.raises(ImageProviderError, match="HUGGINGFACE_TOKEN"):
        create_image_provider(
            Settings(
                _env_file=None,
                image_provider="huggingface",
                image_model_name="owner/open-image-model",
            )
        )

    provider = create_image_provider(
        Settings(
            _env_file=None,
            image_provider="huggingface",
            image_model_name="owner/open-image-model",
            huggingface_token=SecretStr("hf-placeholder"),
        )
    )
    assert isinstance(provider, HuggingFaceImageProvider)
