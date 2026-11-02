"""Pure helpers for copyable image-prompt blocks in article Markdown."""

import re
import unicodedata

from seo_blog_assistant.schemas import ImagePlaceholder

_HEADER_PATTERN = re.compile(
    r"(?m)^> 🖼️ \*\*IMAGE PLACEHOLDER \((img_[1-4])\):[^\r\n]*\*\*$"
)
_BLOCK_PATTERN = re.compile(
    r"(?m)^> 🖼️ \*\*IMAGE PLACEHOLDER \((img_[1-4])\):[^\r\n]*\*\*"
    r"(?:\r?\n>[^\r\n]*)*"
)
_BACKTICK_RUN = re.compile(r"`+")
_WHITESPACE = re.compile(r"\s+")


def sanitize_image_prompt(prompt: str) -> str:
    """Collapse whitespace and remove Markdown-breaking or control characters."""
    without_controls = "".join(
        character if unicodedata.category(character) != "Cc" else " "
        for character in prompt
    )
    return _WHITESPACE.sub(" ", _BACKTICK_RUN.sub("", without_controls)).strip()


def shorten_alt_text(text: str, max_length: int = 120) -> str:
    """Shorten text at a word boundary without exceeding the requested length."""
    normalized = sanitize_image_prompt(text)
    if len(normalized) <= max_length:
        return normalized
    shortened = normalized[:max_length].rsplit(" ", 1)[0].rstrip()
    return shortened or normalized[:max_length]


def build_placeholder_block(image_id: str, prompt: str, alt_text: str) -> str:
    """Build a copyable, blockquoted Markdown placeholder."""
    if not re.fullmatch(r"img_[1-4]", image_id):
        raise ValueError("image_id must be img_1 through img_4")

    safe_prompt = sanitize_image_prompt(prompt)
    safe_alt_text = shorten_alt_text(alt_text)
    if not safe_prompt or not safe_alt_text:
        raise ValueError("prompt and alt_text must contain visible text")
    quoted_block = "\n".join(
        [
            f"> 🖼️ **IMAGE PLACEHOLDER ({image_id}): place an image here**",
            ">",
            "> Copy this prompt into your image generator, then replace this whole block with the image:",
            ">",
            "> ```text",
            f"> {safe_prompt}",
            "> ```",
            ">",
            f"> Alt text: {safe_alt_text}",
        ]
    )
    return f"\n\n{quoted_block}\n\n"


def split_placeholder_blocks(markdown: str) -> list[tuple[str, str | None]]:
    """Split Markdown into text and complete placeholder-block segments."""
    segments: list[tuple[str, str | None]] = []
    position = 0
    for match in _BLOCK_PATTERN.finditer(markdown):
        segments.append((markdown[position : match.start()], None))
        segments.append((match.group(0), match.group(1)))
        position = match.end()
    segments.append((markdown[position:], None))
    return segments


def strip_image_placeholders(markdown: str) -> str:
    """Remove full image-placeholder blockquotes and their added blank lines."""
    output = markdown
    for block, _image_id in reversed(split_placeholder_blocks(markdown)):
        if _image_id is None:
            continue
        start = output.find(block)
        if start < 0:
            continue
        end = start + len(block)
        if start >= 2 and output[start - 2 : start] == "\n\n":
            start -= 2
        if output[end : end + 2] == "\n\n":
            end += 2
        output = output[:start] + output[end:]
    return output


def create_image_placeholder(
    *,
    image_id: str,
    after_heading: str,
    prompt: str,
) -> ImagePlaceholder:
    """Create a validated placeholder record with normalized prompt and alt text."""
    safe_prompt = sanitize_image_prompt(prompt)
    return ImagePlaceholder(
        image_id=image_id,
        after_heading=after_heading,
        prompt=safe_prompt,
        alt_text=shorten_alt_text(safe_prompt),
    )
