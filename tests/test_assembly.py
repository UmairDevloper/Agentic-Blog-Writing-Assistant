import pytest
from pydantic import ValidationError

from seo_blog_assistant.schemas import (
    ArticleBookends,
    ArticleResult,
    DraftSubsection,
    ImageGenerationResult,
    ResearchBundle,
    SEOPlan,
    SectionDraft,
)
from seo_blog_assistant.workflow.assembly import assemble_article


def _plan() -> SEOPlan:
    return SEOPlan(
        title="A practical topic guide",
        meta_description="A useful overview with practical takeaways.",
        search_intent="Informational",
        primary_keyword="topic guide",
        related_keyword_ideas=["topic basics"],
        target_words=300,
        introduction_brief="Set context for the article.",
        introduction_word_budget=100,
        sections=[
            {
                "id": f"section_{position + 1}",
                "position": position,
                "heading": f"Section {position + 1}",
                "purpose": "Explain a distinct part of the topic.",
                "key_points": [f"Explain key idea {position + 1}"],
                "h3_headings": ["How it works"],
                "word_budget": 50,
            }
            for position in range(3)
        ],
        conclusion_brief="Summarize the main takeaways.",
        conclusion_word_budget=50,
    )


def _draft(position: int, source_ids: list[str] | None = None) -> SectionDraft:
    return SectionDraft(
        task_id=f"section_{position + 1}",
        position=position,
        heading=f"Section {position + 1}",
        opening="Start with the main idea.",
        subsections=[
            DraftSubsection(
                heading="How it works",
                content=f"Explain key idea {position + 1}.",
                source_ids=source_ids or [],
            )
        ],
        closing_insight="Connect the idea to the reader's goal.",
        key_points_covered=[f"Explain key idea {position + 1}"],
    )


def _research() -> ResearchBundle:
    return ResearchBundle(
        search_performed=True,
        provider="test",
        searched_at="2026-10-30T12:00:00Z",
        search_queries=["topic evidence"],
        sources=[
            {
                "source_id": "source_1",
                "url": "https://example.com/guide",
                "title": "A verified guide",
                "publisher": "example.com",
                "published_at": "2026-10-01",
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_1",
                "source_id": "source_1",
                "text": "Retrieved evidence.",
            }
        ],
    )


def test_assembly_is_deterministic_and_orders_sections_by_plan() -> None:
    plan = _plan()
    result = assemble_article(
        plan=plan,
        bookends=ArticleBookends(
            introduction="Introduce the topic and what readers will learn.",
            conclusion="Review the most important takeaways.",
        ),
        section_drafts=[_draft(2), _draft(0), _draft(1)],
    )

    assert result.markdown.startswith("# A practical topic guide\n")
    assert result.markdown.index("## Section 1") < result.markdown.index("## Section 2")
    assert result.markdown.index("## Section 2") < result.markdown.index("## Section 3")
    assert result.markdown.endswith(
        "## Conclusion\n\nReview the most important takeaways.\n"
    )
    assert result.sources == []
    assert result.warnings == []
    assert result.image_result.status == "skipped"
    assert assemble_article(
        plan=plan,
        bookends=ArticleBookends(
            introduction="Introduce the topic and what readers will learn.",
            conclusion="Review the most important takeaways.",
        ),
        section_drafts=[_draft(0), _draft(1), _draft(2)],
    ).markdown == result.markdown


def test_assembly_preserves_research_metadata_and_renders_valid_citations() -> None:
    result = assemble_article(
        plan=_plan(),
        bookends=ArticleBookends(
            introduction="Introduce the topic and what readers will learn.",
            conclusion="Review the most important takeaways.",
        ),
        section_drafts=[_draft(0, ["source_1"]), _draft(1), _draft(2)],
        research=_research(),
    )

    assert len(result.sources) == 1
    assert result.sources[0].publisher == "example.com"
    assert result.sources[0].published_at == "2026-10-01"
    assert "Explain key idea 1. [source_1]" in result.markdown
    assert (
        "- [A verified guide](https://example.com/guide) [source_1]"
        in result.markdown
    )
    assert result.image_result.status == "skipped"


@pytest.mark.parametrize(
    ("drafts", "research", "match"),
    [
        ([_draft(0), _draft(1)], None, "exactly one draft"),
        ([_draft(0), _draft(1), _draft(1)], None, "section identities"),
    ],
)
def test_assembly_rejects_incomplete_or_untraceable_drafts(
    drafts: list[SectionDraft],
    research: ResearchBundle | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        assemble_article(
            plan=_plan(),
            bookends=ArticleBookends(
                introduction="Introduce the topic.",
                conclusion="Review the takeaways.",
            ),
            section_drafts=drafts,
            research=research,
        )


def test_assembly_accepts_inline_known_citation_without_declared_source_id() -> None:
    drafts = [_draft(0), _draft(1), _draft(2)]
    drafts[0] = drafts[0].model_copy(
        update={
            "subsections": [
                DraftSubsection(
                    heading="How it works",
                    content="Explain the process. [source_1]",
                )
            ]
        }
    )

    result = assemble_article(
        plan=_plan(),
        bookends=ArticleBookends(
            introduction="Introduce the topic.",
            conclusion="Review the takeaways.",
        ),
        section_drafts=drafts,
        research=_research(),
    )

    assert "[source_1]" in result.markdown
    assert [source.source_id for source in result.sources] == ["source_1"]


def test_assembly_removes_unknown_citations_and_carries_warnings() -> None:
    drafts = [_draft(0, ["missing_source"]), _draft(1), _draft(2)]
    drafts[0] = drafts[0].model_copy(
        update={
            "subsections": [
                DraftSubsection(
                    heading="How it works",
                    content="Explain the process. [source_missing]",
                    source_ids=["missing_source"],
                )
            ]
        }
    )

    result = assemble_article(
        plan=_plan(),
        bookends=ArticleBookends(
            introduction="Introduce the topic.",
            conclusion="Review the takeaways.",
        ),
        section_drafts=drafts,
        research=_research(),
        warnings=["A section was repaired."],
    )

    assert "[source_missing]" not in result.markdown
    assert result.sources == []
    assert "A section was repaired." in result.warnings
    assert any("removed citations" in warning for warning in result.warnings)
    assert any("removed an inline citation" in warning for warning in result.warnings)


def test_article_schema_rejects_unknown_markdown_citation() -> None:
    with pytest.raises(ValidationError, match="listed sources"):
        ArticleResult(
            title="A practical topic guide",
            meta_description="A useful overview.",
            search_intent="Informational",
            primary_keyword="topic guide",
            related_keyword_ideas=[],
            markdown="# A practical topic guide\n\n[source_1]\n",
            sources=[],
            image_result=ImageGenerationResult(status="skipped"),
        )
