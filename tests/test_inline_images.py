import asyncio
import base64
from io import BytesIO

from PIL import Image

from seo_blog_assistant.schemas import (
    ArticleResult,
    GeneratedImage,
    ImageGenerationResult,
)
from seo_blog_assistant.providers.image import ImageProviderError
from seo_blog_assistant.workflow.inline_images import (
    generate_inline_images,
    parse_inline_image_markers,
)


def _generated_image(alt_text: str) -> GeneratedImage:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(buffer, format="PNG")
    return GeneratedImage(
        model_name="owner/open-image-model",
        prompt="Editorial illustration, no text.",
        alt_text=alt_text,
        image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def test_marker_parser_tracks_headings_and_drops_markers_over_limit() -> None:
    markdown = (
        "# Article\n"
        "## First section\n"
        "{{IMAGE: A detailed mountain landscape illustration}}\n"
        "## Second section\n"
        "{{IMAGE: A colorful geometric garden illustration}}\n"
        "## Third section\n"
        "{{IMAGE: A clean modern kitchen illustration}}\n"
    )

    parsed = parse_inline_image_markers(
        markdown,
        generate_images=True,
        max_images=2,
    )

    assert [marker.image_id for marker in parsed.markers] == ["img_1", "img_2"]
    assert [marker.after_heading for marker in parsed.markers] == [
        "First section",
        "Second section",
    ]
    assert "{{IMAGE:" not in parsed.markdown
    assert len(parsed.warnings) == 1
    assert "Dropped 1" in parsed.warnings[0]


def test_marker_parser_accepts_descriptions_up_to_400_characters() -> None:
    description = "A" * 400
    parsed = parse_inline_image_markers(
        f"## Section\n{{{{IMAGE: {description}}}}}\n",
        generate_images=True,
        max_images=1,
    )

    assert len(parsed.markers) == 1
    assert parsed.markers[0].description == description
    assert parsed.warnings == []


def test_marker_parser_drops_markers_not_immediately_under_section_heading() -> None:
    parsed = parse_inline_image_markers(
        "## Section\nParagraph before image.\n"
        "{{IMAGE: A detailed editorial scene description}}\n",
        generate_images=True,
        max_images=1,
    )

    assert parsed.markers == []
    assert "{{IMAGE:" not in parsed.markdown
    assert "Dropped 1" in parsed.warnings[0]


def test_markers_are_removed_when_disabled_and_urls_are_rejected() -> None:
    markdown = (
        "## Section\n"
        "{{IMAGE: A detailed illustrative scene}}\n"
        "{{IMAGE: A visual using https://example.com/image}}\n"
        "{{IMAGE: too short}}\n"
    )

    disabled = parse_inline_image_markers(
        markdown,
        generate_images=False,
        max_images=2,
    )
    url_rejected = parse_inline_image_markers(
        markdown,
        generate_images=True,
        max_images=2,
    )

    assert disabled.markers == []
    assert "{{IMAGE:" not in disabled.markdown
    assert "Dropped 3" in disabled.warnings[0]
    assert [marker.description for marker in url_rejected.markers] == [
        "A detailed illustrative scene"
    ]
    assert "{{IMAGE:" not in url_rejected.markdown
    assert "Dropped 2" in url_rejected.warnings[0]


def test_inline_images_fail_independently_and_validate_article_markdown() -> None:
    class MixedImageProvider:
        async def generate_image(self, prompt: str, alt_text: str) -> GeneratedImage:
            if "night sky" in alt_text:
                raise ImageProviderError(
                    "Image generation failed (model=test, provider=auto): "
                    "Hugging Face monthly credits exhausted"
                )
            return _generated_image(alt_text)

    markdown = (
        "# A practical topic guide\n\n"
        "## Energy options\n"
        "{{IMAGE: A bright solar panel on a clear day}}\n\n"
        "Useful article content.\n"
        "## Night sky\n"
        "{{IMAGE: A detailed starry night sky illustration}}\n"
    )
    result = asyncio.run(
        generate_inline_images(
            markdown,
            generate_images=True,
            max_images=2,
            image_provider=MixedImageProvider(),
        )
    )
    article = ArticleResult(
        title="A practical topic guide",
        meta_description="A useful overview.",
        search_intent="Informational",
        primary_keyword="topic guide",
        related_keyword_ideas=[],
        markdown=result.markdown,
        sources=[],
        image_result=result.image_result,
        inline_images=result.inline_images,
        warnings=result.warnings,
    )

    assert result.image_result.status == "completed"
    assert [image.status for image in article.inline_images] == [
        "completed",
        "failed",
    ]
    assert "![A bright solar panel on a clear day](image:img_1)" in article.markdown
    assert "starry night sky illustration](image:" not in article.markdown
    assert "{{IMAGE:" not in article.markdown
    assert "monthly credits exhausted" in (article.inline_images[1].message or "")
    assert "Some images failed" in (article.image_result.message or "")


def test_all_inline_image_failures_set_failed_aggregate_status() -> None:
    class FailedImageProvider:
        async def generate_image(self, _prompt: str, _alt_text: str) -> GeneratedImage:
            raise ImageProviderError(
                "Image generation failed (model=test, provider=auto): rate limited"
            )

    result = asyncio.run(
        generate_inline_images(
            "# Article\n## Section\n"
            "{{IMAGE: A detailed editorial image of a forest}}\n",
            generate_images=True,
            max_images=2,
            image_provider=FailedImageProvider(),
        )
    )

    assert result.image_result.status == "failed"
    assert "rate limited" in (result.image_result.message or "")
    assert result.inline_images[0].status == "failed"
    assert "rate limited" in (result.inline_images[0].message or "")
    assert "{{IMAGE:" not in result.markdown
    article = ArticleResult(
        title="Article",
        meta_description="A useful overview.",
        search_intent="Informational",
        primary_keyword="topic",
        related_keyword_ideas=[],
        markdown=result.markdown,
        sources=[],
        image_result=result.image_result,
        inline_images=result.inline_images,
        warnings=result.warnings,
    )
    assert article.title == "Article"


def test_missing_marker_generates_contextual_image_when_requested() -> None:
    class RecordingImageProvider:
        prompts: list[str] = []

        async def generate_image(self, prompt: str, alt_text: str) -> GeneratedImage:
            self.prompts.append(prompt)
            return _generated_image(alt_text)

    provider = RecordingImageProvider()
    result = asyncio.run(
        generate_inline_images(
            "# A practical topic guide\n\n"
            "## Introduction\nOpening.\n"
            "## Energy options\nUseful article content.\n",
            generate_images=True,
            max_images=2,
            image_provider=provider,
            fallback_description="Editorial illustration for an energy guide",
        )
    )

    assert result.image_result.status == "completed"
    assert len(provider.prompts) == 1
    assert result.inline_images[0].after_heading == "Energy options"
    assert "![Editorial illustration for an energy guide](image:img_1)" in (
        result.markdown
    )
    assert any("no usable image marker" in warning for warning in result.warnings)


def test_diagram_marker_is_rewritten_to_abstract_scene_and_warned() -> None:
    class RecordingImageProvider:
        prompt = ""

        async def generate_image(self, prompt: str, alt_text: str) -> GeneratedImage:
            self.prompt = prompt
            return _generated_image(alt_text)

    provider = RecordingImageProvider()
    result = asyncio.run(
        generate_inline_images(
            "# A practical topic guide\n\n"
            "## Energy options\n"
            "{{IMAGE: Create a flowchart diagram for energy choices}}\n",
            generate_images=True,
            max_images=2,
            image_provider=provider,
        )
    )
    article = ArticleResult(
        title="A practical topic guide",
        meta_description="A useful overview.",
        search_intent="Informational",
        primary_keyword="topic",
        related_keyword_ideas=[],
        markdown=result.markdown,
        sources=[],
        image_result=result.image_result,
        inline_images=result.inline_images,
        warnings=result.warnings,
    )

    assert "Abstract editorial illustration representing: Energy options" in (
        provider.prompt
    )
    assert "flowchart diagram" not in provider.prompt
    assert "Clean editorial illustration, flat colors, no text." in provider.prompt
    assert "Rewrote a diagram-like image description" in article.warnings[0]
    assert article.inline_images[0].alt_text.startswith(
        "Abstract editorial illustration"
    )


def test_no_markers_result_is_skipped() -> None:
    result = asyncio.run(
        generate_inline_images(
            "# Article\n## Section\nNo image requested.",
            generate_images=True,
            max_images=0,
            image_provider=None,
        )
    )

    assert result.image_result.status == "skipped"
    assert result.inline_images == []
