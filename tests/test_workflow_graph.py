import asyncio
import json
from typing import TypeVar

import pytest
from pydantic import BaseModel

from seo_blog_assistant.config import Settings
from seo_blog_assistant.providers.search import ResearchProviderError
from seo_blog_assistant.schemas import (
    ArticleBookends,
    BlogRequest,
    ResearchBundle,
    ResearchDecision,
    SEOPlan,
    SectionDraft,
)
from seo_blog_assistant.workflow.topic_profiles import TopicProfile
from seo_blog_assistant.providers.text_model import TextResult
from seo_blog_assistant.workflow.graph import (
    build_blog_workflow_graph,
    build_research_planning_graph,
)

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

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> TextResult:
        raise AssertionError("Research-planning graph does not write articles.")


class FakeResearchProvider:
    def __init__(self) -> None:
        self.requests: list[BlogRequest] = []

    async def search_and_retrieve(
        self,
        request: BlogRequest,
        _profile: TopicProfile,
    ) -> ResearchBundle:
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


def _decision(mode: str, *, category: str = "general") -> ResearchDecision:
    return ResearchDecision(
        selected_mode=mode,
        rationale="Selected for this graph test.",
        estimated_knowledge_sufficient=True,
        confidence=0.8,
        topic_category=category,
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


class CountingArticleModel:
    def __init__(
        self,
        *,
        research_mode: str = "model_knowledge",
        emit_image_marker: bool = False,
    ) -> None:
        self.research_mode = research_mode
        self.emit_image_marker = emit_image_marker
        self.structured_schemas: list[type[BaseModel]] = []
        self.text_calls: list[tuple[str, int]] = []

    async def generate_structured(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        self.structured_schemas.append(schema)
        if schema is ResearchDecision:
            response = _decision(self.research_mode)
        elif schema is SEOPlan:
            request_payload = json.loads(user_prompt)["request"]
            response = _plan_for_target(request_payload["target_words"])
        elif schema is SectionDraft:
            section = json.loads(user_prompt)["section"]
            response = {
                "task_id": section["id"],
                "position": section["position"],
                "heading": section["heading"],
                "opening": "Start with a useful idea for the reader.",
                "subsections": [
                    {
                        "heading": heading,
                        "content": "Explain the planned concept with a clear example.",
                    }
                    for heading in section["h3_headings"]
                ],
                "closing_insight": "Connect the section to the reader's goal.",
                "key_points_covered": section["key_points"],
            }
        elif schema is ArticleBookends:
            response = {
                "introduction": "Introduce the topic and what readers will learn.",
                "conclusion": "Review the most important practical takeaways.",
            }
        else:
            raise AssertionError(f"Unexpected schema: {schema.__name__}")
        return schema.model_validate(response)

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> TextResult:
        payload = json.loads(user_prompt)
        self.text_calls.append((user_prompt, max_tokens))
        lines: list[str] = []
        if "introduction" in payload:
            lines.extend(
                [
                    f"# {payload['title']}",
                    "## Introduction",
                    "Introduce the requested topic for readers.",
                ]
            )
        for section in payload["sections"]:
            lines.append(f"## {section['heading']}")
            if self.emit_image_marker and "Add at most" in system_prompt:
                lines.append(
                    "{{IMAGE: A warm editorial illustration of a thoughtful reader "
                    "exploring a bright landscape with layered hills, soft golden "
                    "light, calm colors, a wide balanced composition, and a hopeful "
                    "mood, with clear foreground depth and gentle natural textures}}"
                )
            lines.extend(
                [
                    f"### {section['h3_headings'][0]}",
                    "Explain the key idea with an example.",
                ]
            )
            for heading in section["h3_headings"][1:]:
                lines.extend(
                    [
                        f"### {heading}",
                        "Give readers a second helpful consideration.",
                    ]
                )
        if "conclusion" in payload:
            lines.extend(
                [
                    "## Conclusion",
                    "Summarize the article's key takeaways.",
                ]
            )
        return TextResult(text="\n\n".join(lines), truncated=False)


def _plan_for_target(target_words: int) -> SEOPlan:
    if target_words == 1200:
        return _plan()
    section_count = 6
    return SEOPlan(
        title="A practical topic guide",
        meta_description="A useful overview with practical takeaways.",
        search_intent="Informational",
        primary_keyword="topic guide",
        related_keyword_ideas=[],
        target_words=target_words,
        introduction_brief="Set context for the article.",
        introduction_word_budget=150,
        sections=[
            {
                "id": f"section_{position + 1}",
                "position": position,
                "heading": f"Section {position + 1}",
                "purpose": "Explain a distinct part of the topic.",
                "key_points": ["Explain the key idea"],
                "h3_headings": ["How it works"],
                "word_budget": 450,
            }
            for position in range(section_count)
        ],
        conclusion_brief="Summarize the main takeaways.",
        conclusion_word_budget=150,
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


def test_full_graph_selects_batched_and_per_section_node_paths() -> None:
    model = FakeStructuredTextModel()
    placeholder_settings = Settings(_env_file=None, image_mode="placeholder")
    generate_settings = Settings(_env_file=None, image_mode="generate")
    batched_nodes = {
        node_id
        for node_id in build_blog_workflow_graph(
            model,
            settings=placeholder_settings,
        ).get_graph().nodes
        if node_id not in {"__start__", "__end__"}
    }
    per_section_nodes = {
        node_id
        for node_id in build_blog_workflow_graph(
            model,
            writing_mode="per_section",
            settings=placeholder_settings,
        )
        .get_graph()
        .nodes
        if node_id not in {"__start__", "__end__"}
    }

    assert {
        "write_article",
        "prepare_image_placeholders",
        "assemble_batched",
    } <= batched_nodes
    assert not {"write_section", "collect_sections", "assemble_article"} & batched_nodes
    assert {
        "write_section",
        "collect_sections",
        "prepare_image_placeholders",
        "assemble_article",
    } <= per_section_nodes
    assert not {"write_article", "generate_inline_images", "images"} & per_section_nodes
    generate_nodes = {
        node_id
        for node_id in build_blog_workflow_graph(
            model,
            settings=generate_settings,
        ).get_graph().nodes
        if node_id not in {"__start__", "__end__"}
    }
    assert "generate_inline_images" in generate_nodes


@pytest.mark.parametrize(
    ("target_words", "expected_calls"),
    [(1200, 3), (3000, 4)],
)
def test_batched_graph_uses_three_or_four_text_calls(
    target_words: int,
    expected_calls: int,
) -> None:
    model = CountingArticleModel()
    graph = build_blog_workflow_graph(
        model,
        max_words_per_call=1500,
    )

    result = asyncio.run(
        graph.ainvoke(
            {"request": BlogRequest(topic="A useful topic", target_words=target_words)}
        )
    )

    assert len(model.structured_schemas) == 2
    assert len(model.text_calls) == expected_calls - 2
    assert len(model.structured_schemas) + len(model.text_calls) == expected_calls
    assert result["article"].title == "A practical topic guide"
    assert result["article"].research_mode == "model_knowledge"


def test_per_section_graph_keeps_its_existing_text_call_pattern() -> None:
    model = CountingArticleModel()
    graph = build_blog_workflow_graph(model, writing_mode="per_section")

    asyncio.run(
        graph.ainvoke({"request": BlogRequest(topic="A useful topic")})
    )

    assert model.structured_schemas == [
        ResearchDecision,
        SEOPlan,
        SectionDraft,
        SectionDraft,
        SectionDraft,
        ArticleBookends,
    ]
    assert model.text_calls == []


def test_sports_article_uses_research_without_adding_text_model_calls() -> None:
    model = CountingArticleModel(research_mode="model_knowledge")
    graph = build_blog_workflow_graph(
        model,
        search_provider=FakeResearchProvider(),
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "request": BlogRequest(
                    topic="Harbor City football",
                    category="sports",
                )
            }
        )
    )

    assert len(model.structured_schemas) + len(model.text_calls) == 3
    assert result["article"].research_mode == "web_research"
    assert result["article"].topic_category == "sports"


def test_required_category_without_search_provider_has_category_error() -> None:
    model = FakeStructuredTextModel(_decision("model_knowledge", category="sports"))
    graph = build_blog_workflow_graph(model)

    with pytest.raises(
        ResearchProviderError,
        match="Sports articles need web research for stats and records",
    ):
        asyncio.run(graph.ainvoke({"request": BlogRequest(topic="A sports topic")}))


def test_recommended_category_without_provider_warns_in_final_article() -> None:
    model = CountingArticleModel()
    graph = build_blog_workflow_graph(model)

    result = asyncio.run(
        graph.ainvoke(
            {
                "request": BlogRequest(
                    topic="Cloud computing basics",
                    category="technology",
                )
            }
        )
    )

    assert result["article"].research_mode == "model_knowledge"
    assert any(
        "research was recommended" in warning.casefold()
        and "figures are unverified" in warning
        for warning in result["article"].warnings
    )


def test_research_categories_warn_about_figures_missing_from_writer_evidence() -> None:
    class UnsupportedFigureModel(CountingArticleModel):
        async def generate_text(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            max_tokens: int,
        ) -> TextResult:
            response = await super().generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            return TextResult(
                text=f"{response.text}\n\nThe result was 45% in 2024.",
                truncated=response.truncated,
            )

    model = UnsupportedFigureModel()
    graph = build_blog_workflow_graph(
        model,
        search_provider=FakeResearchProvider(),
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "request": BlogRequest(
                    topic="Harbor City football",
                    category="sports",
                )
            }
        )
    )

    warnings = result["article"].warnings
    assert any("45%, 2024" in warning for warning in warnings)
    assert "45%, 2024" in result["article"].warnings[-1]


def test_batched_research_branch_and_optional_image_failure_keep_article() -> None:
    class FailedImageProvider:
        async def generate_image(
            self,
            _prompt: str,
            _alt_text: str,
        ) -> object:
            raise RuntimeError("private image provider detail")

    model = CountingArticleModel(
        research_mode="web_research",
        emit_image_marker=True,
    )
    graph = build_blog_workflow_graph(
        model,
        search_provider=FakeResearchProvider(),
        image_provider=FailedImageProvider(),
        max_images=1,
        settings=Settings(_env_file=None, image_mode="generate"),
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "request": BlogRequest(
                    topic="A current useful topic",
                    generate_images=True,
                )
            }
        )
    )

    assert len(model.structured_schemas) == 2
    assert len(model.text_calls) == 1
    assert result["article"].title == "A practical topic guide"
    assert result["article"].research_mode == "web_research"
    assert result["article"].image_result.status == "failed"
    assert result["article"].inline_images[0].status == "failed"
    assert "private image provider detail" not in result["article"].inline_images[0].message
    assert "{{IMAGE:" not in result["article"].markdown


def test_batched_placeholder_mode_never_calls_image_provider() -> None:
    class FailIfCalledImageProvider:
        async def generate_image(self, _prompt: str, _alt_text: str) -> object:
            raise AssertionError("placeholder mode must not call image inference")

    model = CountingArticleModel(emit_image_marker=True)
    graph = build_blog_workflow_graph(
        model,
        image_provider=FailIfCalledImageProvider(),
        max_images=1,
        settings=Settings(_env_file=None, image_mode="placeholder"),
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "request": BlogRequest(
                    topic="A useful topic",
                    generate_images=True,
                )
            }
        )
    )

    article = result["article"]
    assert article.image_result.status == "skipped"
    assert article.inline_images == []
    assert [item.image_id for item in article.image_placeholders] == ["img_1"]
    assert article.image_placeholders[0].prompt in article.markdown
    assert "IMAGE PLACEHOLDER (img_1)" in article.markdown
    assert "{{IMAGE:" not in article.markdown
    assert "fallback prompt" not in " ".join(article.warnings)


def test_batched_placeholder_mode_creates_the_contextual_fallback_prompt() -> None:
    model = CountingArticleModel()
    graph = build_blog_workflow_graph(
        model,
        max_images=1,
        settings=Settings(_env_file=None, image_mode="placeholder"),
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "request": BlogRequest(
                    topic="A useful topic",
                    generate_images=True,
                )
            }
        )
    )

    article = result["article"]
    assert len(article.image_placeholders) == 1
    assert article.image_placeholders[0].after_heading == "Section 1"
    assert article.image_placeholders[0].prompt == (
        "Editorial illustration representing Section 1 in an article about "
        "A practical topic guide. Clean composition, soft lighting, flat colors, "
        "no text, no logos."
    )
    assert any("fallback prompt was created" in warning for warning in article.warnings)


def test_off_mode_has_no_image_generation_node_or_markers() -> None:
    model = CountingArticleModel(emit_image_marker=True)
    graph = build_blog_workflow_graph(
        model,
        max_images=1,
        settings=Settings(_env_file=None, image_mode="off"),
    )
    nodes = set(graph.get_graph().nodes)
    result = asyncio.run(
        graph.ainvoke(
            {
                "request": BlogRequest(
                    topic="A useful topic",
                    generate_images=True,
                )
            }
        )
    )

    assert "generate_inline_images" not in nodes
    assert result["article"].image_placeholders == []
    assert result["article"].inline_images == []
    assert "{{IMAGE:" not in result["article"].markdown


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
