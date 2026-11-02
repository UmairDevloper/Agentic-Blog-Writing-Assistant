from seo_blog_assistant.workflow.fact_guard import find_unsupported_figures
from seo_blog_assistant.workflow.image_placeholders import build_placeholder_block


def test_fact_guard_accepts_figures_in_evidence_and_request() -> None:
    markdown = (
        "# Results\n\nThe study included 1,200 people and found a 45% change. "
        "The trial ran in 2024 and the budget was $3.2 million."
    )

    unsupported = find_unsupported_figures(
        markdown,
        ["The study included 1200 people, a 45.0% change, and $3.2 million in 2024."],
        "The requested season is 2025.",
    )

    assert unsupported == []


def test_fact_guard_reports_invented_percent_year_and_grouped_number_once() -> None:
    markdown = (
        "# Report\n\nThe rate rose to 45% in 2024, affecting 12,500 people. "
        "A code has 999 digits."
    )

    assert find_unsupported_figures(markdown, ["The study ran in 2023."]) == [
        "45%",
        "2024",
        "12,500",
        "999",
    ]


def test_fact_guard_ignores_code_urls_placeholders_and_structural_counts() -> None:
    placeholder = build_placeholder_block(
        "img_1",
        "A figure showing 99% improvement in 2022 with warm editorial lighting",
        "Prompt",
    )
    markdown = (
        "# 1. Article\n\n1. First item\n\n## 2. Results\n\n"
        "The target is 1200 words. See https://example.com/2024/45-percent.\n\n"
        "```text\nA fake result is 85% in 2020.\n```\n"
        f"{placeholder}"
    )

    assert find_unsupported_figures(markdown, []) == []


def test_fact_guard_normalizes_decimal_zeroes_and_reports_word_scale() -> None:
    markdown = "The market is worth 3.20 million and rose 17.5%."

    assert find_unsupported_figures(markdown, ["Market value: 3.2 million."]) == [
        "17.5%"
    ]
