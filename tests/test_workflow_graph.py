import asyncio
from typing import TypeVar

import pytest
from pydantic import BaseModel

from seo_blog_assistant.providers.search import ResearchProviderError
from seo_blog_assistant.schemas import (
    BlogRequest,
    ResearchBundle,
    ResearchDecision,
    SEOPlan,
)
from seo_blog_assistant.workflow.graph import build_research_planning_graph

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class FakeStructuredTextModel:
    def __init__(self, *responses: BaseModel) -> None:
        self.responses = list(responses)
        self.calls: list[type[BaseModel]] = []

    async def generate_structured(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        self.calls.append(schema)
        response = self.responses.pop(0)
        return schema.model_validate(response)


class FakeResearchProvider:
    def __init__(self) -> None:
        self.requests: list[BlogRequest] = []

    async def search_and_retrieve(self, request: BlogRequest) -> ResearchBundle:
        self.requests.append(request)
        return ResearchBundle(
            search_performed=True,
            provider="fake",
            searched_at="2026-10-30T12:00:00Z",
            search_queries=[request.topic],
            sources=[
                {
                    "source_id": "source_1",
                    "url": "https://example.com/guide",
                    "title": "A verified guide",
                }
            ],
            evidence=[
                {
                    "evidence_id": "evidence_1",
                    "source_id": "source_1",
                    "text": "Evidence from a retrieved page.",
                }
            ],
        )


def _decision(mode: str) -> ResearchDecision:
    return ResearchDecision(
        selected_mode=mode,
        rationale="Selected for this graph test.",
        estimated_knowledge_sufficient=True,
        confidence=0.8,
    )


def _plan() -> SEOPlan:
    return SEOPlan(
        title="A practical topic guide",
        meta_description="A useful overview with practical takeaways.",
        search_intent="Informational",
        primary_keyword="topic guide",
        related_keyword_ideas=[],
        target_words=1200,
        introduction_brief="Set context for the article.",
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
        conclusion_brief="Summarize the main takeaways.",
        conclusion_word_budget=100,
    )


def test_graph_routes_model_knowledge_to_seo_planning() -> None:
    model = FakeStructuredTextModel(_decision("model_knowledge"), _plan())
    graph = build_research_planning_graph(model)

    result = asyncio.run(
        graph.ainvoke({"request": BlogRequest(topic="A general topic")})
    )

    assert result["workflow_stage"] == "plan_ready"
    assert result["research_decision"].selected_mode == "model_knowledge"
    assert result["seo_plan"].target_words == 1200
    assert model.calls == [ResearchDecision, SEOPlan]


def test_graph_rejects_research_route_without_configured_provider() -> None:
    model = FakeStructuredTextModel(_decision("web_research"))
    graph = build_research_planning_graph(model)

    with pytest.raises(ResearchProviderError, match="no search provider"):
        asyncio.run(graph.ainvoke({"request": BlogRequest(topic="A current topic")}))

    assert model.calls == [ResearchDecision]


def test_graph_searches_then_plans_with_evidence_and_status_updates() -> None:
    model = FakeStructuredTextModel(_decision("web_research"), _plan())
    search_provider = FakeResearchProvider()
    graph = build_research_planning_graph(model, search_provider)

    async def collect_updates() -> list[dict[str, object]]:
        return [
            update
            async for update in graph.astream(
                {"request": BlogRequest(topic="A current topic")},
                stream_mode="updates",
            )
        ]

    updates = asyncio.run(collect_updates())

    start_update = next(update["start_research"] for update in updates if "start_research" in update)
    research_update = next(update["research"] for update in updates if "research" in update)
    plan_update = next(
        update["create_seo_plan"] for update in updates if "create_seo_plan" in update
    )
    assert start_update["workflow_stage"] == "research_started"
    assert research_update["workflow_stage"] == "research_completed"
    assert research_update["research_bundle"].sources[0].source_id == "source_1"
    assert plan_update["workflow_stage"] == "plan_ready"
    assert search_provider.requests[0].topic == "A current topic"
    assert model.calls == [ResearchDecision, SEOPlan]
