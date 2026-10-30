"""Image-provider interface and Hugging Face hosted inference adapter."""

import asyncio
import base64
from io import BytesIO
from typing import Protocol

import httpx
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from pydantic import SecretStr, ValidationError

from seo_blog_assistant.config import Settings
from seo_blog_assistant.schemas import GeneratedImage


class ImageProviderError(RuntimeError):
    """Raised when an image provider cannot produce a valid image."""


class ImageProvider(Protocol):
    """Generates a validated illustration independently of text generation."""

    async def generate_image(self, prompt: str, alt_text: str) -> GeneratedImage: ...


class ImageInferenceClient(Protocol):
    """Synchronous hosted inference client surface used by the adapter."""

    def text_to_image(self, prompt: str) -> Image.Image: ...


class HuggingFaceImageProvider:
    """Generate PNG images through Hugging Face's hosted inference API."""

    def __init__(
        self,
        model_name: str,
        token: SecretStr,
        *,
        timeout_seconds: int = 120,
        client: ImageInferenceClient | None = None,
    ) -> None:
        normalized_model = model_name.strip()
        if not normalized_model:
            raise ValueError("Hugging Face image model name must not be empty")
        if not token.get_secret_value():
            raise ValueError("Hugging Face token must not be empty")
        if not 1 <= timeout_seconds <= 600:
            raise ValueError("timeout_seconds must be between 1 and 600")

        self._model_name = normalized_model
        self._client = client or InferenceClient(
            model=normalized_model,
            provider="hf-inference",
            token=token.get_secret_value(),
            timeout=timeout_seconds,
        )

    async def generate_image(self, prompt: str, alt_text: str) -> GeneratedImage:
        if not prompt.strip() or len(prompt) > 2000:
            raise ValueError("image prompt must contain 1 to 2000 characters")
        if not alt_text.strip() or len(alt_text) > 500:
            raise ValueError("image alt text must contain 1 to 500 characters")

        try:
            generated = await asyncio.to_thread(
                self._client.text_to_image, prompt.strip()
            )
            if not isinstance(generated, Image.Image):
                raise ImageProviderError(
                    "Hugging Face returned an unsupported image response."
                )
            if (
                generated.width > 8192
                or generated.height > 8192
                or generated.width * generated.height > 40_000_000
            ):
                raise ImageProviderError(
                    "Hugging Face returned an image with unsupported dimensions."
                )

            buffer = BytesIO()
            generated.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            if len(image_data) > 15_000_000:
                raise ImageProviderError(
                    "Hugging Face returned an image that exceeds the 15 MB limit."
                )
            return GeneratedImage(
                model_name=self._model_name,
                prompt=prompt.strip(),
                alt_text=alt_text.strip(),
                image_base64=base64.b64encode(image_data).decode("ascii"),
            )
        except (HfHubHTTPError, httpx.HTTPError, TimeoutError, OSError):
            raise ImageProviderError(
                "Hugging Face image generation failed."
            ) from None
        except ValidationError:
            raise ImageProviderError(
                "Hugging Face returned an invalid image response."
            ) from None


def create_image_provider(settings: Settings) -> ImageProvider:
    """Create the configured image provider or raise actionable guidance."""
    if settings.image_provider != "huggingface":
        raise ImageProviderError(
            "Configure SEO_BLOG_IMAGE_PROVIDER=huggingface to enable image generation."
        )
    if settings.image_model_name is None or not settings.image_model_name.strip():
        raise ImageProviderError(
            "Set SEO_BLOG_IMAGE_MODEL_NAME to enable image generation."
        )
    if settings.huggingface_token is None:
        raise ImageProviderError(
            "Set SEO_BLOG_HUGGINGFACE_TOKEN to enable image generation."
        )
    return HuggingFaceImageProvider(
        settings.image_model_name,
        settings.huggingface_token,
        timeout_seconds=settings.image_timeout_seconds,
    )
