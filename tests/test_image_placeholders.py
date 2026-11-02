import pytest

from seo_blog_assistant.workflow.image_placeholders import (
    build_placeholder_block,
    sanitize_image_prompt,
    shorten_alt_text,
    strip_image_placeholders,
)


def test_placeholder_block_has_the_exact_copyable_format() -> None:
    prompt = (
        "A small sailboat crossing a quiet lake at dawn, editorial illustration, "
        "soft blue colors, peaceful mood, wide composition."
    )
    expected = (
        "\n\n"
        "> 🖼️ **IMAGE PLACEHOLDER (img_1): place an image here**\n"
        ">\n"
        "> Copy this prompt into your image generator, then replace this whole block with the image:\n"
        ">\n"
        "> ```text\n"
        f"> {prompt}\n"
        "> ```\n"
        ">\n"
        f"> Alt text: {shorten_alt_text(prompt)}\n\n"
    )

    assert build_placeholder_block("img_1", prompt, prompt) == expected


def test_placeholder_sanitizes_backticks_control_characters_and_long_alt_text() -> None:
    prompt = "A quiet `lake`\nwith soft light and `no` garbled symbols"
    block = build_placeholder_block("img_2", prompt, "word " * 40)

    assert "A quiet lake with soft light and no garbled symbols" in block
    assert "`lake`" not in block
    assert len(shorten_alt_text("word " * 40)) <= 120
    assert shorten_alt_text("word " * 40).endswith("word")
    assert sanitize_image_prompt("one\u0000two") == "one two"


def test_placeholder_ids_are_validated() -> None:
    with pytest.raises(ValueError, match="img_1 through img_4"):
        build_placeholder_block("img_5", "A scene", "A scene")


def test_stripping_placeholder_blocks_round_trips_surrounding_article_text() -> None:
    original = "# Article\n\n## A section\nBefore.\nAfter.\n"
    block = build_placeholder_block(
        "img_1",
        "A calm illustrated shoreline under a lavender sunrise.",
        "A calm illustrated shoreline",
    )
    markdown = original.replace("After.", f"{block}After.")

    assert strip_image_placeholders(markdown) == original


def test_stripping_removes_multiple_complete_blocks() -> None:
    first = build_placeholder_block("img_1", "A first illustrative scene", "first")
    second = build_placeholder_block("img_2", "A second illustrative scene", "second")
    markdown = f"# Article\n\n## One\n{first}## Two\n{second}"

    assert strip_image_placeholders(markdown) == "# Article\n\n## One\n## Two\n"
