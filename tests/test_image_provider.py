import asyncio
import base64
from io import BytesIO

import pytest
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
        self.prompts: list[str] = []

    def text_to_image(self, prompt: str) -> Image.Image:
        self.prompts.append(prompt)
        return _png_image()


def test_huggingface_provider_returns_validated_png_without_network() -> None:
    client = FakeInferenceClient()
    provider = HuggingFaceImageProvider(
        "owner/open-image-model",
        SecretStr("hf-test-secret"),
        timeout_seconds=30,
        client=client,
    )

    result = asyncio.run(
        provider.generate_image("Editorial illustration", "A blue square")
    )

    assert client.prompts == ["Editorial illustration"]
    assert result.model_name == "owner/open-image-model"
    assert result.media_type == "image/png"
    image_data = base64.b64decode(result.image_base64, validate=True)
    with Image.open(BytesIO(image_data)) as image:
        assert image.format == "PNG"
        assert image.size == (2, 2)


def test_huggingface_provider_sanitizes_hosted_inference_errors() -> None:
    class FailedInferenceClient:
        def text_to_image(self, _prompt: str) -> Image.Image:
            raise TimeoutError("request failed with hf-private-token")

    provider = HuggingFaceImageProvider(
        "owner/open-image-model",
        SecretStr("hf-private-token"),
        client=FailedInferenceClient(),
    )

    with pytest.raises(ImageProviderError, match="generation failed") as error:
        asyncio.run(provider.generate_image("An image", "An image"))

    assert "hf-private-token" not in str(error.value)


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
