import asyncio
from typing import TypeVar

import pytest
from pydantic import BaseModel

from seo_blog_assistant.schemas import (
    BlogRequest,
    ResearchBundle,
    ResearchDecision,
    SEOPlan,
)
from seo_blog_assistant.workflow.planning import (
    SEO_PLAN_SYSTEM_PROMPT,
    create_seo_plan,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class FakeStructuredTextModel:
    def __init__(self, response: BaseModel) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def generate_structured(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        self.calls.append(
            {
                "schema": schema,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return schema.model_validate(self.response)


def _plan(target_words: int = 1200) -> SEOPlan:
    return SEOPlan(
        title="A practical topic guide",
        meta_description="A useful overview with practical takeaways.",
        search_intent="Informational",
        primary_keyword="topic guide",
        related_keyword_ideas=["topic basics"],
        target_words=target_words,
        introduction_brief="Set context and explain what the reader will learn.",
        introduction_word_budget=100,
        sections=[
            {
                "id": f"section_{position + 1}",
                "position": position,
                "heading": f"Section {position + 1}",
                "purpose": "Explain a distinct part of the topic.",
                "key_points": ["Explain the key idea"],
                "h3_headings": ["How it works"],
                "word_budget": 300,
            }
            for position in range(3)
        ],
        conclusion_brief="Recap the key points for the reader.",
        conclusion_word_budget=100,
    )


def _decision(mode: str) -> ResearchDecision:
    return ResearchDecision(
        selected_mode=mode,
        rationale="Selected for this test.",
        estimated_knowledge_sufficient=True,
        confidence=0.8,
    )


def _research_bundle() -> ResearchBundle:
    return ResearchBundle(
        search_performed=True,
        provider="tavily",
        searched_at="2026-10-30T12:00:00Z",
        search_queries=["topic current guidance"],
        sources=[
            {
                "source_id": "source_1",
                "url": "https://example.com/guidance",
                "title": "Current topic guidance",
            }
        ],
        evidence=[
            {
                "evidence_id": "evidence_1",
                "source_id": "source_1",
                "text": "The source provides current guidance.",
            }
        ],
    )


def test_model_knowledge_plan_is_a_brief_not_finished_article_prose() -> None:
    model = FakeStructuredTextModel(_plan())
    request = BlogRequest(topic="A useful topic", target_words=1200)

    plan = asyncio.run(create_seo_plan(request, _decision("model_knowledge"), model))

    assert plan.target_words == request.target_words
    assert [section.position for section in plan.sections] == [0, 1, 2]
    assert "not finished article prose" in SEO_PLAN_SYSTEM_PROMPT
    assert len(model.calls) == 1
    assert '"research_mode": "model_knowledge"' in str(model.calls[0]["user_prompt"])


def test_research_plan_requires_a_completed_search_bundle() -> None:
    model = FakeStructuredTextModel(_plan())

    with pytest.raises(ValueError, match="completed ResearchBundle"):
        asyncio.run(create_seo_plan(
            BlogRequest(topic="A useful topic"),
            _decision("web_research"),
            model,
        ))

    assert model.calls == []


def test_research_plan_receives_evidence_from_completed_search() -> None:
    model = FakeStructuredTextModel(_plan())

    plan = asyncio.run(
        create_seo_plan(
            BlogRequest(topic="A useful topic"),
            _decision("web_research"),
            model,
            research=_research_bundle(),
        )
    )

    assert plan.title == "A practical topic guide"
    assert '"source_id": "source_1"' in str(model.calls[0]["user_prompt"])
    assert '"evidence_id": "evidence_1"' in str(model.calls[0]["user_prompt"])


def test_plan_must_preserve_requested_article_length() -> None:
    model = FakeStructuredTextModel(_plan(target_words=1000))

    with pytest.raises(ValueError, match="must match the requested"):
        asyncio.run(
            create_seo_plan(
                BlogRequest(topic="A useful topic", target_words=1200),
                _decision("model_knowledge"),
                model,
            )
        )
