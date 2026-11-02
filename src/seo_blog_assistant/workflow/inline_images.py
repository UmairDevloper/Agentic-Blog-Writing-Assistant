"""Parse generated image markers and isolate optional image inference."""

import asyncio
import logging
import re
from dataclasses import dataclass

from seo_blog_assistant.providers.image import ImageProvider, ImageProviderError
from seo_blog_assistant.schemas import (
    GeneratedImage,
    ImageGenerationResult,
    ImagePlaceholder,
    InlineImage,
    SEOPlan,
    SectionDraft,
)
from seo_blog_assistant.workflow.image_placeholders import (
    build_placeholder_block,
    create_image_placeholder,
)

logger = logging.getLogger(__name__)
_MARKER_PATTERN = re.compile(r"^\{\{IMAGE:\s*(.{10,400}?)\s*\}\}$")
_MARKER_CANDIDATE = re.compile(r"^\s*\{\{IMAGE:")
_LEVEL_TWO_HEADING = re.compile(r"^##[ \t]+(.+?)\s*$")
_DIAGRAM_WORDS = re.compile(
    r"\b(?:diagram|flowchart|chart|graph|screenshot|table)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedImageMarker:
    """Validated marker extracted from Markdown."""

    image_id: str
    after_heading: str
    description: str
    token: str


@dataclass(frozen=True)
class ParsedImageContent:
    """Markdown with retained markers replaced by private tokens."""

    markdown: str
    markers: list[ParsedImageMarker]
    warnings: list[str]


@dataclass
class InlineImageGeneration:
    """Markdown and image outcomes returned by the optional image stage."""

    markdown: str
    inline_images: list[InlineImage]
    image_result: ImageGenerationResult
    warnings: list[str]


@dataclass
class ImagePlaceholderGeneration:
    """Markdown and copyable prompts produced without image inference."""

    markdown: str
    image_placeholders: list[ImagePlaceholder]
    image_result: ImageGenerationResult
    warnings: list[str]


def parse_inline_image_markers(
    markdown: str,
    *,
    generate_images: bool,
    max_images: int,
) -> ParsedImageContent:
    """Keep only valid, bounded markers under article section headings."""
    if not 0 <= max_images <= 4:
        raise ValueError("max_images must be between 0 and 4")

    lines = markdown.splitlines(keepends=True)
    current_heading: str | None = None
    previous_line_was_heading = False
    markers: list[ParsedImageMarker] = []
    dropped_count = 0

    for line_index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        heading_match = _LEVEL_TWO_HEADING.fullmatch(body)
        if heading_match:
            current_heading = heading_match.group(1).strip()
            previous_line_was_heading = True
            continue
        if not _MARKER_CANDIDATE.match(body):
            if body.strip():
                previous_line_was_heading = False
            continue

        marker_match = _MARKER_PATTERN.fullmatch(body)
        description = marker_match.group(1).strip() if marker_match else ""
        if (
            not generate_images
            or len(markers) >= max_images
            or marker_match is None
            or not 10 <= len(description) <= 400
            or current_heading is None
            or not previous_line_was_heading
            or "http" in description.casefold()
        ):
            lines[line_index] = _line_ending(line)
            dropped_count += 1
            previous_line_was_heading = False
            continue

        image_id = f"img_{len(markers) + 1}"
        token = f"\x00SEO_BLOG_IMAGE_{image_id}\x00"
        markers.append(
            ParsedImageMarker(
                image_id=image_id,
                after_heading=current_heading,
                description=description,
                token=token,
            )
        )
        lines[line_index] = token + _line_ending(line)
        previous_line_was_heading = False

    warnings = (
        [
            f"Dropped {dropped_count} invalid, disabled, URL-bearing, or over-limit "
            "image marker(s)."
        ]
        if dropped_count
        else []
    )
    return ParsedImageContent(
        markdown="".join(lines),
        markers=markers,
        warnings=warnings,
    )


def build_image_placeholders(
    markdown: str,
    *,
    title: str,
    generate_images: bool,
    max_images: int,
) -> ImagePlaceholderGeneration:
    """Replace usable image markers with copyable blocks without inference."""
    parsed = parse_inline_image_markers(
        markdown,
        generate_images=generate_images,
        max_images=max_images,
    )
    warnings = list(parsed.warnings)
    if not parsed.markers and generate_images and max_images > 0:
        heading = _first_article_section_heading(parsed.markdown)
        if heading is not None:
            fallback_prompt = (
                f"Editorial illustration representing {heading} in an article "
                f"about {title}. Clean composition, soft lighting, flat colors, "
                "no text, no logos."
            )
            fallback = _fallback_marker(parsed.markdown, fallback_prompt)
            if fallback is not None:
                parsed = ParsedImageContent(
                    markdown=fallback[0],
                    markers=[fallback[1]],
                    warnings=parsed.warnings,
                )
                warnings.append(
                    "The writer returned no usable image marker; a fallback "
                    "prompt was created."
                )

    placeholders = [
        create_image_placeholder(
            image_id=marker.image_id,
            after_heading=marker.after_heading,
            prompt=marker.description,
        )
        for marker in parsed.markers
    ]
    final_markdown = parsed.markdown
    for marker, placeholder in zip(parsed.markers, placeholders):
        block = build_placeholder_block(
            placeholder.image_id,
            placeholder.prompt,
            placeholder.alt_text,
        )
        final_markdown = final_markdown.replace(marker.token, block[1:-1])

    message = (
        "Image placeholders were added; generate the images with an external tool "
        "and replace the blocks."
        if placeholders
        else "Image generation was not requested."
        if not generate_images
        else "Image placeholders are disabled by the configured image limit."
        if max_images == 0
        else "No valid image markers were found."
    )
    return ImagePlaceholderGeneration(
        markdown=final_markdown,
        image_placeholders=placeholders,
        image_result=ImageGenerationResult(status="skipped", message=message),
        warnings=warnings,
    )


def build_section_image_placeholders(
    *,
    plan: SEOPlan,
    section_drafts: list[SectionDraft],
    generate_images: bool,
    max_images: int,
) -> ImagePlaceholderGeneration:
    """Build ordered placeholders from structured per-section image prompts."""
    if not 0 <= max_images <= 4:
        raise ValueError("max_images must be between 0 and 4")
    warnings: list[str] = []
    candidates: list[tuple[str, str]] = []
    if generate_images:
        drafts_by_id = {draft.task_id: draft for draft in section_drafts}
        for section in plan.sections:
            prompt = drafts_by_id[section.id].image_prompt
            if prompt is None or not prompt.strip():
                continue
            normalized_prompt = " ".join(prompt.split())
            word_count = len(normalized_prompt.split())
            if (
                "http" in normalized_prompt.casefold()
                or not 25 <= word_count <= 60
                or len(normalized_prompt) > 400
            ):
                warnings.append(
                    f"Section {section.id}: ignored an invalid image prompt."
                )
                continue
            candidates.append((section.heading, normalized_prompt))

    if generate_images and max_images > 0 and not candidates:
        section_heading = plan.sections[0].heading
        candidates.append(
            (
                section_heading,
                f"Editorial illustration representing {section_heading} in an "
                f"article about {plan.title}. Clean composition, soft lighting, "
                "flat colors, no text, no logos.",
            )
        )
        warnings.append(
            "The writer returned no usable image marker; a fallback prompt was created."
        )

    if len(candidates) > max_images:
        warnings.append(
            f"Dropped {len(candidates) - max_images} image prompt(s) over the "
            "configured article limit."
        )
        candidates = candidates[:max_images]

    placeholders = [
        create_image_placeholder(
            image_id=f"img_{index}",
            after_heading=heading,
            prompt=prompt,
        )
        for index, (heading, prompt) in enumerate(candidates, start=1)
    ]
    message = (
        "Image placeholders were added; generate the images with an external tool "
        "and replace the blocks."
        if placeholders
        else "Image generation was not requested."
        if not generate_images
        else "Image placeholders are disabled by the configured image limit."
        if max_images == 0
        else "No valid image markers were found."
    )
    return ImagePlaceholderGeneration(
        markdown="",
        image_placeholders=placeholders,
        image_result=ImageGenerationResult(status="skipped", message=message),
        warnings=warnings,
    )


async def generate_inline_images(
    markdown: str,
    *,
    generate_images: bool,
    max_images: int,
    image_provider: ImageProvider | None,
    fallback_description: str | None = None,
) -> InlineImageGeneration:
    """Generate retained markers independently with bounded concurrency."""
    parsed = parse_inline_image_markers(
        markdown,
        generate_images=generate_images,
        max_images=max_images,
    )
    warnings = list(parsed.warnings)
    if (
        not parsed.markers
        and generate_images
        and max_images > 0
        and fallback_description is not None
    ):
        fallback = _fallback_marker(parsed.markdown, fallback_description)
        if fallback is not None:
            parsed = ParsedImageContent(
                markdown=fallback[0],
                markers=[fallback[1]],
                warnings=parsed.warnings,
            )
            warnings.append(
                "The writer returned no usable image marker; generated one "
                "contextual image instead."
            )

    if not parsed.markers:
        message = (
            "Image generation was not requested."
            if not generate_images
            else "Image generation is disabled by the configured image limit."
            if max_images == 0
            else "No valid image markers were found."
        )
        return InlineImageGeneration(
            markdown=parsed.markdown,
            inline_images=[],
            image_result=ImageGenerationResult(status="skipped", message=message),
            warnings=warnings,
        )

    semaphore = asyncio.Semaphore(2)

    async def generate(marker: ParsedImageMarker) -> InlineImage:
        description = marker.description
        if _DIAGRAM_WORDS.search(description):
            description = (
                f"Abstract editorial illustration representing: "
                f"{marker.after_heading}"
            )[:200]
            warnings.append(
                "Rewrote a diagram-like image description as an abstract "
                f"illustration for '{marker.after_heading[:100]}'."
            )
        alt_text = description[:500]
        if image_provider is None:
            message = (
                "Image provider is not configured; set "
                "SEO_BLOG_IMAGE_PROVIDER, SEO_BLOG_IMAGE_MODEL_NAME, and "
                "SEO_BLOG_HUGGINGFACE_TOKEN."
            )
            return InlineImage(
                image_id=marker.image_id,
                after_heading=marker.after_heading,
                alt_text=alt_text,
                status="failed",
                message=message[:500],
            )

        prompt = (
            f"{description}. Illustrative or abstract scene and visual "
            "metaphor; not a diagram, chart, flowchart, screenshot, or UI. "
            "Clean editorial illustration, flat colors, no text. No letters, "
            "numbers, logos or watermarks."
        )[:2000]
        try:
            async with semaphore:
                generated = await image_provider.generate_image(prompt, alt_text)
            image = GeneratedImage.model_validate(generated)
        except ImageProviderError as exc:
            return InlineImage(
                image_id=marker.image_id,
                after_heading=marker.after_heading,
                alt_text=alt_text,
                status="failed",
                message=str(exc)[:500],
            )
        except Exception as exc:
            safe_exception = RuntimeError(type(exc).__name__)
            logger.exception(
                "Unexpected image provider failure image_id=%s",
                marker.image_id,
                exc_info=(RuntimeError, safe_exception, exc.__traceback__),
            )
            return InlineImage(
                image_id=marker.image_id,
                after_heading=marker.after_heading,
                alt_text=alt_text,
                status="failed",
                message=(
                    "Image generation failed unexpectedly; article text remains "
                    "available."
                ),
            )
        return InlineImage(
            image_id=marker.image_id,
            after_heading=marker.after_heading,
            alt_text=alt_text,
            status="completed",
            image=image,
        )

    images = await asyncio.gather(*(generate(marker) for marker in parsed.markers))
    replacements = {
        marker.image_id: image
        for marker, image in zip(parsed.markers, images)
    }
    final_markdown = parsed.markdown
    for marker in parsed.markers:
        image = replacements[marker.image_id]
        replacement = (
            f"![{_escape_markdown_alt(image.alt_text)}](image:{image.image_id})"
            if image.status == "completed"
            else ""
        )
        final_markdown = final_markdown.replace(marker.token, replacement)

    successful_images = [image for image in images if image.status == "completed"]
    failed_images = [image for image in images if image.status == "failed"]
    failure_details = "; ".join(
        image.message or "Image generation failed." for image in failed_images
    )[:400]
    if successful_images:
        image_result = ImageGenerationResult(
            status="completed",
            image=successful_images[0].image,
            message=(
                f"Some images failed: {failure_details}" if failed_images else None
            ),
        )
    else:
        image_result = ImageGenerationResult(
            status="failed",
            message=(
                "All inline image generations failed; article text is available. "
                f"{failure_details}"
            )[:500],
        )

    return InlineImageGeneration(
        markdown=final_markdown,
        inline_images=images,
        image_result=image_result,
        warnings=warnings,
    )


def _fallback_marker(
    markdown: str,
    description: str,
) -> tuple[str, ParsedImageMarker] | None:
    """Insert one contextual image marker below the first article section."""
    normalized_description = " ".join(description.split())[:400]
    if len(normalized_description) < 10 or "http" in normalized_description.casefold():
        return None

    lines = markdown.splitlines(keepends=True)
    headings = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines)
        if (match := _LEVEL_TWO_HEADING.fullmatch(line.rstrip("\r\n")))
    ]
    if not headings:
        return None
    heading_index, heading = next(
        (
            (index, title)
            for index, title in headings
            if title.casefold() not in {"introduction", "conclusion"}
        ),
        headings[0],
    )
    token = "\x00SEO_BLOG_IMAGE_img_1\x00"
    newline = "\r\n" if lines[heading_index].endswith("\r\n") else "\n"
    if not lines[heading_index].endswith(("\n", "\r")):
        lines[heading_index] += newline
    lines.insert(heading_index + 1, token + newline)
    return "".join(lines), ParsedImageMarker(
        image_id="img_1",
        after_heading=heading,
        description=normalized_description,
        token=token,
    )


def _first_article_section_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        match = _LEVEL_TWO_HEADING.fullmatch(line)
        if match and match.group(1).strip().casefold() not in {
            "introduction",
            "conclusion",
            "sources",
        }:
            return match.group(1).strip()
    return None


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""


def _escape_markdown_alt(alt_text: str) -> str:
    return alt_text.replace("\\", "\\\\").replace("]", "\\]")
