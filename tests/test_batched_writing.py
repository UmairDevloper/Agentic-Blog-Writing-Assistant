import asyncio
import json

import pytest

from seo_blog_assistant.config import Settings
from seo_blog_assistant.providers.text_model import TextResult
from seo_blog_assistant.schemas import (
    ArticleResult,
    BlogRequest,
    ImageGenerationResult,
    ResearchBundle,
    SEOPlan,
)
from seo_blog_assistant.workflow.batched_writing import (
    build_writing_prompt,
    check_pass_output,
    split_plan_into_passes,
    write_article_passes,
)
from seo_blog_assistant.workflow.topic_profiles import get_profile


def _plan(
    *,
    target_words: int = 1200,
    section_budgets: list[int] | None = None,
) -> SEOPlan:
    budgets = section_budgets or [300, 300, 300]
    introduction_budget = 150
    conclusion_budget = 150
    return SEOPlan(
        title="A practical topic guide",
        meta_description="A useful overview with practical takeaways.",
        search_intent="Informational",
        primary_keyword="topic guide",
        related_keyword_ideas=[f"related keyword {index}" for index in range(8)],
        target_words=target_words,
        introduction_brief="Introduce the topic and explain what readers will learn.",
        introduction_word_budget=introduction_budget,
        sections=[
            {
                "id": f"section_{index + 1}",
                "position": index,
                "heading": f"Section {index + 1}",
                "purpose": "Explain a distinct part of the topic.",
                "key_points": ["Explain the key idea", "Give a practical example"],
                "h3_headings": ["How it works", "What to consider"],
                "word_budget": budget,
            }
            for index, budget in enumerate(budgets)
        ],
        conclusion_brief="Summarize the practical takeaways.",
        conclusion_word_budget=conclusion_budget,
    )


def _markdown(plan: SEOPlan, *, reverse_sections: bool = False) -> str:
    sections = list(plan.sections)
    if reverse_sections:
        sections.reverse()
    parts = ["# Model title", "## Introduction", "An introduction."]
    for section in sections:
        parts.extend(
            [
                f"## {section.heading}",
                f"### {section.h3_headings[0]}",
                "Explain the key idea with a useful practical example.",
                f"### {section.h3_headings[1]}",
                "Add a second useful explanation for the reader.",
            ]
        )
    parts.extend(["## Conclusion", "A concise conclusion."])
    return "\n\n".join(parts)


class QueuedTextModel:
    def __init__(self, *responses: TextResult) -> None:
        self.responses = list(responses)
        self.prompts: list[tuple[str, str, int]] = []

    async def generate_structured(self, **_kwargs: object) -> object:
        raise AssertionError("Batched writing must use plain-text generation.")

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> TextResult:
        self.prompts.append((system_prompt, user_prompt, max_tokens))
        if not self.responses:
            raise AssertionError("No fake text response remains.")
        return self.responses.pop(0)


def test_split_plan_uses_one_pass_for_1200_and_two_contiguous_passes_for_3000() -> None:
    small_passes = split_plan_into_passes(_plan(), max_words_per_call=1500)
    large_plan = _plan(target_words=3000, section_budgets=[450] * 6)
    large_passes = split_plan_into_passes(large_plan, max_words_per_call=1500)

    assert len(small_passes) == 1
    assert small_passes[0].include_introduction
    assert small_passes[0].include_conclusion
    assert len(large_passes) == 2
    assert large_passes[0].include_introduction
    assert not large_passes[0].include_conclusion
    assert not large_passes[1].include_introduction
    assert large_passes[1].include_conclusion
    assert [
        section.id
        for writing_pass in large_passes
        for section in writing_pass.sections
    ] == [f"section_{index}" for index in range(1, 7)]


def test_writing_prompt_is_compact_and_conditionally_includes_image_rules() -> None:
    plan = _plan()
    writing_pass = split_plan_into_passes(plan, 1500)[0]
    research = ResearchBundle(
        search_performed=True,
        provider="tavily",
        searched_at="2026-10-30T12:00:00Z",
        search_queries=["topic sources"],
        sources=[
            {
                "source_id": "source_verified",
                "url": "https://example.com/guide",
                "title": "A verified guide",
                "publisher": "example.com",
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_1",
                "source_id": "source_verified",
                "text": "Useful research evidence for this guide.",
            }
        ],
    )
    request = BlogRequest(topic="A useful topic", generate_images=True)
    system_prompt, user_prompt = build_writing_prompt(
        request=request,
        plan=plan,
        writing_pass=writing_pass,
        research=research,
        remaining_image_allowance=2,
    )
    payload = json.loads(user_prompt)

    assert "null" not in user_prompt
    assert "audience" not in payload
    assert "target_country" not in payload
    assert payload["related_keywords"] == plan.related_keyword_ideas[:6]
    assert payload["research"]["sources"] == [
        {"label": "S1", "title": "A verified guide"}
    ]
    assert payload["research"]["evidence"][0]["source_id"] == "S1"
    assert "example.com" not in user_prompt
    assert "{{IMAGE: ready-to-use image prompt}}" in system_prompt
    assert "25-60 words" in system_prompt
    assert "no URLs, real people, brand names, logos, or readable text" in system_prompt
    assert "clean infographic style, simple shapes, minimal labels" in system_prompt
    assert "Write no image markers." in build_writing_prompt(
        request=BlogRequest(topic="A useful topic"),
        plan=plan,
        writing_pass=writing_pass,
    )[0]
    assert "Write no image markers." in build_writing_prompt(
        request=request,
        plan=plan,
        writing_pass=writing_pass,
        remaining_image_allowance=0,
    )[0]


def test_batched_prompt_includes_only_selected_category_and_fact_rule() -> None:
    plan = _plan()
    writing_pass = split_plan_into_passes(plan, 1500)[0]
    system_prompt, user_prompt = build_writing_prompt(
        request=BlogRequest(topic="A sports topic"),
        plan=plan,
        writing_pass=writing_pass,
        profile=get_profile("sports"),
        content_angle="season preview with historical comparison",
    )

    assert "Category: Sports" in user_prompt
    assert "Category: Food" not in user_prompt
    assert "season preview with historical comparison" in user_prompt
    assert "Never guess statistics, dates, scores, records, polls, or quotes" in (
        system_prompt
    )
def test_missing_heading_retries_once_with_validation_feedback() -> None:
    plan = _plan()
    incomplete = _markdown(plan).replace("## Section 2\n\n", "")
    model = QueuedTextModel(
        TextResult(text=incomplete, truncated=False),
        TextResult(text=_markdown(plan), truncated=False),
    )

    article = asyncio.run(
        write_article_passes(
            request=BlogRequest(topic="A useful topic"),
            plan=plan,
            model=model,
            settings=Settings(_env_file=None),
        )
    )

    assert len(model.prompts) == 2
    assert "validation_feedback" in json.loads(model.prompts[1][1])
    assert model.prompts[0][2] == 2300
    assert article.markdown.startswith("# A practical topic guide\n")
    assert "## Section 2" in article.markdown


def test_missing_heading_after_retry_fails_with_heading_name() -> None:
    plan = _plan()
    incomplete = _markdown(plan).replace("## Section 2\n\n", "")
    model = QueuedTextModel(
        TextResult(text=incomplete, truncated=False),
        TextResult(text=incomplete, truncated=False),
    )

    with pytest.raises(ValueError, match="Section 2"):
        asyncio.run(
            write_article_passes(
                request=BlogRequest(topic="A useful topic"),
                plan=plan,
                model=model,
                settings=Settings(_env_file=None),
            )
        )


def test_truncated_pass_continues_using_only_short_tail_and_remaining_headings() -> None:
    plan = _plan()
    first_part = "# Model title\n\n## Introduction\n\n" + ("Opening words. " * 120)
    continuation_parts = []
    for section in plan.sections:
        continuation_parts.extend(
            [
                f"## {section.heading}",
                f"### {section.h3_headings[0]}",
                "Explain the key idea with a useful practical example.",
                f"### {section.h3_headings[1]}",
                "Add a second useful explanation for the reader.",
            ]
        )
    continuation_parts.extend(["## Conclusion", "A concise conclusion."])
    continuation = "\n\n".join(continuation_parts)
    model = QueuedTextModel(
        TextResult(text=first_part, truncated=True),
        TextResult(text=continuation, truncated=False),
    )

    article = asyncio.run(
        write_article_passes(
            request=BlogRequest(topic="A useful topic"),
            plan=plan,
            model=model,
            settings=Settings(_env_file=None),
        )
    )

    continuation_payload = json.loads(model.prompts[1][1])
    assert len(model.prompts) == 2
    assert len(continuation_payload["last_text"]) <= 300
    assert continuation_payload["headings_to_write"] == [
        "Section 1",
        "Section 2",
        "Section 3",
        "Conclusion",
    ]
    assert article.markdown.startswith("# A practical topic guide\n")


def test_pass_check_warns_for_wrong_order_extra_heading_and_word_count() -> None:
    plan = _plan()
    writing_pass = split_plan_into_passes(plan, 1500)[0]
    text = _markdown(plan, reverse_sections=True) + "\n\n## Unexpected heading\nShort."

    check = check_pass_output(text, writing_pass)

    assert not check.headings_in_order
    assert check.unexpected_headings == ["Unexpected heading"]
    assert not check.word_count_in_range


def test_writer_warns_but_keeps_out_of_order_and_short_passes() -> None:
    plan = _plan()
    model = QueuedTextModel(
        TextResult(
            text=_markdown(plan, reverse_sections=True) + "\n\n## Extra\nShort.",
            truncated=False,
        )
    )

    article = asyncio.run(
        write_article_passes(
            request=BlogRequest(topic="A useful topic"),
            plan=plan,
            model=model,
            settings=Settings(_env_file=None),
        )
    )

    assert len(model.prompts) == 1
    assert any("out of the planned order" in warning for warning in article.warnings)
    assert any("unexpected or duplicate" in warning for warning in article.warnings)
    assert any("outside the expected range" in warning for warning in article.warnings)


def test_writer_maps_known_source_labels_and_drops_unknown_labels() -> None:
    plan = _plan()
    research = ResearchBundle(
        search_performed=True,
        provider="tavily",
        searched_at="2026-10-30T12:00:00Z",
        search_queries=["topic sources"],
        sources=[
            {
                "source_id": "source_first",
                "url": "https://example.com/first",
                "title": "First source",
            },
            {
                "source_id": "source_second",
                "url": "https://example.com/second",
                "title": "Second source",
            },
        ],
        evidence=[
            {
                "evidence_id": "evidence_1",
                "source_id": "source_first",
                "text": "First source evidence.",
            },
            {
                "evidence_id": "evidence_2",
                "source_id": "source_second",
                "text": "Second source evidence.",
            },
        ],
    )
    markdown = _markdown(plan).replace(
        "Explain the key idea with a useful practical example.",
        "A supported claim [S1], raw source ID [source_second], "
        "and unsupported label [S9].",
        1,
    )
    model = QueuedTextModel(TextResult(text=markdown, truncated=False))

    written = asyncio.run(
        write_article_passes(
            request=BlogRequest(topic="A useful topic", require_citations=True),
            plan=plan,
            model=model,
            settings=Settings(_env_file=None),
            research=research,
        )
    )
    article = ArticleResult(
        title=plan.title,
        meta_description=plan.meta_description,
        search_intent=plan.search_intent,
        primary_keyword=plan.primary_keyword,
        related_keyword_ideas=plan.related_keyword_ideas,
        markdown=written.markdown,
        sources=written.sources,
        image_result=ImageGenerationResult(status="skipped"),
        warnings=written.warnings,
    )

    assert "[source_first]" in article.markdown
    assert "[source_second]" not in article.markdown
    assert "[S9]" not in article.markdown
    assert len(article.sources) == 1
    assert article.sources[0].source_id == "source_first"
    assert "First source" in article.markdown
    assert "Second source" not in article.markdown
    assert any("unknown research label [S9]" in warning for warning in article.warnings)
    assert any("only supplied S-label citations" in warning for warning in article.warnings)


def test_uncited_research_sources_are_not_added_to_the_article() -> None:
    plan = _plan()
    research = ResearchBundle(
        search_performed=True,
        provider="tavily",
        searched_at="2026-10-30T12:00:00Z",
        search_queries=["topic sources"],
        sources=[
            {
                "source_id": "source_uncited",
                "url": "https://example.com/uncited",
                "title": "Uncited source",
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_1",
                "source_id": "source_uncited",
                "text": "Retrieved but unused source evidence.",
            }
        ],
    )
    model = QueuedTextModel(
        TextResult(text=_markdown(plan), truncated=False)
    )

    written = asyncio.run(
        write_article_passes(
            request=BlogRequest(topic="A useful topic"),
            plan=plan,
            model=model,
            settings=Settings(_env_file=None),
            research=research,
        )
    )

    assert written.sources == []
    assert "## Sources" not in written.markdown
