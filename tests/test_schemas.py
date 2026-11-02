import pytest
from pydantic import ValidationError

from seo_blog_assistant.schemas import (
    BlogRequest,
    ResearchBundle,
    ResearchDecision,
    SEOPlan,
    SectionDraft,
)


def test_blog_request_applies_defaults_and_accepts_settings() -> None:
    request = BlogRequest(
        topic="Practical home energy savings",
        audience="Homeowners",
        tone="friendly",
        target_country="Canada",
        target_words=1800,
        seo_goals=["answer search intent", "use clear headings"],
        require_freshness=True,
        require_citations=True,
        generate_images=True,
    )

    assert request.target_words == 1800
    assert request.require_freshness
    assert request.require_citations
    assert request.generate_images
    assert request.category is None


def test_optional_request_category_and_old_research_response_defaults() -> None:
    request = BlogRequest(topic="A useful topic", category="food")
    decision = ResearchDecision.model_validate(
        {
            "selected_mode": "model_knowledge",
            "rationale": "A general explanation is sufficient.",
            "estimated_knowledge_sufficient": True,
            "confidence": 0.8,
        }
    )

    assert request.category == "food"
    assert decision.topic_category == "general"
    assert decision.content_angle is None


@pytest.mark.parametrize(
    "values",
    [
        {"topic": "  "},
        {"topic": "A", "unexpected": True},
        {"topic": "A valid topic", "target_words": 100},
    ],
)
def test_blog_request_rejects_invalid_input(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BlogRequest.model_validate(values)


def test_research_decision_records_mode_and_estimated_confidence() -> None:
    decision = ResearchDecision(
        selected_mode="model_knowledge",
        rationale="The topic asks for general background rather than current facts.",
        estimated_knowledge_sufficient=True,
        confidence=0.82,
    )

    assert decision.selected_mode == "model_knowledge"
    assert decision.confidence == 0.82


def test_research_decision_rejects_insufficient_model_knowledge_mode() -> None:
    with pytest.raises(ValidationError, match="requires an estimate"):
        ResearchDecision(
            selected_mode="model_knowledge",
            rationale="The topic needs current information.",
            estimated_knowledge_sufficient=False,
            confidence=0.9,
        )


def _planned_section(section_id: str, position: int) -> dict[str, object]:
    return {
        "id": section_id,
        "position": position,
        "heading": f"Section {position + 1}",
        "purpose": "Explain a distinct part of the topic.",
        "key_points": ["Explain the concept"],
        "h3_headings": ["How it works"],
        "word_budget": 250,
    }


def test_seo_plan_rejects_duplicate_section_ids() -> None:
    with pytest.raises(ValidationError, match="section IDs must be unique"):
        SEOPlan(
            title="A useful guide",
            meta_description="A concise guide to the topic.",
            search_intent="Informational",
            primary_keyword="useful guide",
            related_keyword_ideas=[],
            target_words=1200,
            introduction_brief="Introduce the topic and its relevance.",
            introduction_word_budget=100,
            sections=[
                _planned_section("section_1", 0),
                _planned_section("section_1", 1),
                _planned_section("section_3", 2),
            ],
            conclusion_brief="Summarize the key takeaways.",
            conclusion_word_budget=100,
        )


def test_seo_plan_normalizes_model_positions_to_returned_outline_order() -> None:
    plan = SEOPlan(
        title="A useful guide",
        meta_description="A concise guide to the topic.",
        search_intent="Informational",
        primary_keyword="useful guide",
        related_keyword_ideas=[],
        target_words=950,
        introduction_brief="Introduce the topic and its relevance.",
        introduction_word_budget=100,
        sections=[
            _planned_section("section_1", 1),
            _planned_section("section_2", 2),
            _planned_section("section_3", 3),
        ],
        conclusion_brief="Summarize the key takeaways.",
        conclusion_word_budget=100,
    )

    assert [section.id for section in plan.sections] == [
        "section_1",
        "section_2",
        "section_3",
    ]
    assert [section.position for section in plan.sections] == [0, 1, 2]


def test_seo_plan_rescales_budgets_that_are_far_below_target() -> None:
    data = {
        "title": "A useful guide",
        "meta_description": "A concise guide to the topic.",
        "search_intent": "Informational",
        "primary_keyword": "useful guide",
        "related_keyword_ideas": [],
        "target_words": 1000,
        "introduction_brief": "Introduce the topic and its relevance.",
        "introduction_word_budget": 60,
        "sections": [
            {**_planned_section(f"section_{position + 1}", position), "word_budget": 160}
            for position in range(3)
        ],
        "conclusion_brief": "Summarize the key takeaways.",
        "conclusion_word_budget": 60,
    }

    plan = SEOPlan.model_validate(data)

    total_words = (
        plan.introduction_word_budget
        + sum(section.word_budget for section in plan.sections)
        + plan.conclusion_word_budget
    )
    assert total_words == plan.target_words


def test_section_draft_keeps_inline_citation_ids_with_prose() -> None:
    draft = SectionDraft(
        task_id="section_0",
        position=0,
        heading="How the process works",
        opening="Start with the main idea.",
        subsections=[
            {
                "heading": "The key steps",
                "content": "The process has several steps. [source_1]",
                "source_ids": ["source_1"],
            }
        ],
        closing_insight="These steps explain the overall process.",
        key_points_covered=["Describe each step"],
    )

    assert draft.subsections[0].source_ids == ["source_1"]


def test_research_bundle_requires_real_search_marker_and_valid_source_links() -> None:
    bundle = ResearchBundle(
        search_performed=True,
        provider="tavily",
        searched_at="2026-10-30T12:00:00Z",
        search_queries=["topic evidence"],
        sources=[
            {
                "source_id": "source_1",
                "url": "https://example.com/source",
                "title": "A source",
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_1",
                "source_id": "source_1",
                "text": "Evidence from the source.",
            }
        ],
    )

    assert bundle.search_performed is True
    assert bundle.evidence[0].source_id == "source_1"

    with pytest.raises(ValidationError):
        ResearchBundle(
            search_performed=False,
            provider="tavily",
            searched_at="2026-10-30T12:00:00Z",
            search_queries=["topic evidence"],
            sources=[],
            evidence=[],
        )

    with pytest.raises(ValidationError, match="reference a source"):
        ResearchBundle(
            search_performed=True,
            provider="tavily",
            searched_at="2026-10-30T12:00:00Z",
            search_queries=["topic evidence"],
            sources=[
                {
                    "source_id": "source_1",
                    "url": "https://example.com/source",
                    "title": "A source",
                }
            ],
            evidence=[
                {
                    "evidence_id": "evidence_1",
                    "source_id": "missing_source",
                    "text": "Untraceable evidence.",
                }
            ],
        )
