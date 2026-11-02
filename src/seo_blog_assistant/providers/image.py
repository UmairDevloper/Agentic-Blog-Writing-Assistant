"""Image-provider interface and Hugging Face hosted inference adapter."""

import asyncio
import base64
import logging
from io import BytesIO
from typing import Protocol

import httpx
from huggingface_hub import InferenceClient
from PIL import Image
from pydantic import SecretStr, ValidationError

from seo_blog_assistant.config import Settings
from seo_blog_assistant.schemas import GeneratedImage

logger = logging.getLogger(__name__)


class ImageProviderError(RuntimeError):
    """Raised when an image provider cannot produce a valid image."""


class ImageProvider(Protocol):
    """Generates a validated illustration independently of text generation."""

    async def generate_image(self, prompt: str, alt_text: str) -> GeneratedImage: ...


class ImageInferenceClient(Protocol):
    """Synchronous hosted inference client surface used by the adapter."""

    def text_to_image(
        self,
        prompt: str,
        *,
        model: str,
        num_inference_steps: int | None = None,
    ) -> Image.Image: ...


class HuggingFaceImageProvider:
    """Generate PNG images through Hugging Face's hosted inference API."""

    def __init__(
        self,
        model_name: str,
        token: SecretStr,
        *,
        inference_provider: str = "auto",
        num_inference_steps: int | None = 4,
        timeout_seconds: int = 120,
        client: ImageInferenceClient | None = None,
    ) -> None:
        normalized_model = model_name.strip()
        normalized_provider = inference_provider.strip()
        if not normalized_model:
            raise ValueError("Hugging Face image model name must not be empty")
        if not normalized_provider:
            raise ValueError("Hugging Face inference provider must not be empty")
        if not token.get_secret_value():
            raise ValueError("Hugging Face token must not be empty")
        if num_inference_steps is not None and not 1 <= num_inference_steps <= 50:
            raise ValueError("num_inference_steps must be between 1 and 50")
        if not 1 <= timeout_seconds <= 600:
            raise ValueError("timeout_seconds must be between 1 and 600")

        self._model_name = normalized_model
        self._inference_provider = normalized_provider
        self._num_inference_steps = num_inference_steps
        self._client = client or InferenceClient(
            provider=normalized_provider,
            token=token.get_secret_value(),
            timeout=timeout_seconds,
        )

    async def generate_image(self, prompt: str, alt_text: str) -> GeneratedImage:
        if not prompt.strip() or len(prompt) > 2000:
            raise ValueError("image prompt must contain 1 to 2000 characters")
        if not alt_text.strip() or len(alt_text) > 500:
            raise ValueError("image alt text must contain 1 to 500 characters")

        try:
            inference_options: dict[str, object] = {"model": self._model_name}
            if self._num_inference_steps is not None:
                inference_options["num_inference_steps"] = self._num_inference_steps
            generated = await asyncio.to_thread(
                self._client.text_to_image,
                prompt.strip(),
                **inference_options,
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
        except Exception as exc:
            status_code = _response_status_code(exc)
            hint = _safe_failure_hint(exc, status_code)
            logger.warning(
                "image generation failed: model=%s provider=%s type=%s status=%s",
                self._model_name,
                self._inference_provider,
                type(exc).__name__,
                status_code,
            )
            raise ImageProviderError(
                "Image generation failed "
                f"(model={self._model_name[:160]}, "
                f"provider={self._inference_provider[:60]}): {hint}"
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
    try:
        return HuggingFaceImageProvider(
            settings.image_model_name,
            settings.huggingface_token,
            inference_provider=settings.image_inference_provider,
            num_inference_steps=settings.image_num_inference_steps,
            timeout_seconds=settings.image_timeout_seconds,
        )
    except ValueError:
        raise ImageProviderError(
            "Hugging Face image provider settings are invalid."
        ) from None


def _response_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _safe_failure_hint(exc: Exception, status_code: int | None) -> str:
    if status_code in {401, 403}:
        return (
            "token rejected; check SEO_BLOG_HUGGINGFACE_TOKEN has permission "
            "to call Inference Providers"
        )
    if status_code == 402:
        return "Hugging Face monthly credits exhausted"
    if status_code in {404, 422}:
        return (
            "model not available through the selected provider; check "
            "SEO_BLOG_IMAGE_MODEL_NAME and "
            "SEO_BLOG_IMAGE_INFERENCE_PROVIDER"
        )
    if status_code == 429:
        return "rate limited"
    if status_code is not None and 500 <= status_code <= 599:
        return f"provider error ({status_code})"
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or (
        "timeout" in type(exc).__name__.casefold()
    ):
        return "request timed out"
    if isinstance(exc, ValidationError):
        return "provider returned an invalid image"
    return type(exc).__name__[:200]
