import asyncio
import json
from typing import TypeVar

import pytest
from pydantic import BaseModel

from seo_blog_assistant.schemas import BlogRequest, ResearchDecision
from seo_blog_assistant.providers.text_model import TextResult
from seo_blog_assistant.workflow.research import (
    assess_research,
    research_route,
    should_search_web,
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

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> TextResult:
        raise AssertionError("Research assessment only uses structured output.")


def _decision(
    mode: str = "model_knowledge",
    *,
    category: str = "general",
    content_angle: str | None = None,
) -> ResearchDecision:
    return ResearchDecision(
        selected_mode=mode,
        rationale="A general draft appears feasible.",
        estimated_knowledge_sufficient=True,
        confidence=0.75,
        topic_category=category,
        content_angle=content_angle,
    )


def test_research_assessment_uses_model_choice_for_general_topic() -> None:
    model = FakeStructuredTextModel(_decision())
    request = BlogRequest(topic="How rainbows form")

    decision = asyncio.run(assess_research(request, model))

    assert decision.selected_mode == "model_knowledge"
    assert research_route(decision) == "model_knowledge"
    assert not should_search_web(decision)
    assert len(model.calls) == 1
    assert "not proof" in str(model.calls[0]["system_prompt"])
    prompt_payload = json.loads(str(model.calls[0]["user_prompt"]))
    assert prompt_payload["topic"] == request.topic
    assert "audience" not in prompt_payload
    assert "target_country" not in prompt_payload
    assert "seo_goals" not in prompt_payload
    assert "null" not in str(model.calls[0]["user_prompt"])


@pytest.mark.parametrize(
    ("request_values", "required_reason"),
    [
        ({"require_freshness": True}, "fresh information"),
        ({"require_citations": True}, "source citations"),
    ],
)
def test_explicit_research_requirements_override_model_only_choice(
    request_values: dict[str, bool],
    required_reason: str,
) -> None:
    model = FakeStructuredTextModel(_decision())
    request = BlogRequest(topic="A topic with requirements", **request_values)

    decision = asyncio.run(assess_research(request, model))

    assert decision.selected_mode == "web_research"
    assert should_search_web(decision)
    assert required_reason in decision.rationale
    assert "web search" not in decision.rationale.lower()


def test_model_can_choose_web_research_without_explicit_requirement() -> None:
    model = FakeStructuredTextModel(_decision(mode="web_research"))
    request = BlogRequest(topic="A fast-changing technology")

    decision = asyncio.run(assess_research(request, model))

    assert research_route(decision) == "web_research"
    assert should_search_web(decision)


@pytest.mark.parametrize("category", ["sports", "politics"])
def test_required_categories_force_web_research(category: str) -> None:
    model = FakeStructuredTextModel(_decision(category=category))

    decision = asyncio.run(
        assess_research(
            BlogRequest(topic="A topic"),
            model,
            search_provider_configured=True,
        )
    )

    assert decision.topic_category == category
    assert decision.selected_mode == "web_research"


def test_recommended_technology_stays_model_knowledge_without_provider() -> None:
    model = FakeStructuredTextModel(_decision(category="technology"))

    decision = asyncio.run(
        assess_research(
            BlogRequest(topic="A technology topic"),
            model,
            search_provider_configured=False,
        )
    )

    assert decision.selected_mode == "model_knowledge"
    assert "no search provider is configured" in decision.rationale


def test_explicit_category_overrides_model_category_and_angle() -> None:
    model = FakeStructuredTextModel(
        _decision(category="technology", content_angle="cloud deployment trends")
    )

    decision = asyncio.run(
        assess_research(
            BlogRequest(topic="How to cook dumplings", category="food"),
            model,
            search_provider_configured=False,
        )
    )

    assert decision.topic_category == "food"
    assert decision.content_angle == "Food article about How to cook dumplings"
