import asyncio
import base64
import json
from io import BytesIO
from typing import TypeVar

import pytest
from pydantic import BaseModel
from PIL import Image

from seo_blog_assistant.providers.image import ImageProviderError
from seo_blog_assistant.schemas import (
    ArticleBookends,
    BlogRequest,
    GeneratedImage,
    ResearchDecision,
    SEOPlan,
    SectionDraft,
)
from seo_blog_assistant.workflow.graph import build_blog_workflow_graph
from seo_blog_assistant.workflow.writing import write_section

SchemaT = TypeVar("SchemaT", bound=BaseModel)


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
                "key_points": [f"Explain key idea {position + 1}"],
                "h3_headings": ["How it works"],
                "word_budget": 300,
            }
            for position in range(3)
        ],
        conclusion_brief="Summarize the main takeaways.",
        conclusion_word_budget=100,
    )


def _draft(section: dict[str, object]) -> SectionDraft:
    return SectionDraft(
        task_id=str(section["id"]),
        position=int(section["position"]),
        heading=str(section["heading"]),
        opening="Start with the main idea.",
        subsections=[
            {"heading": "How it works", "content": "Explain the key process."}
        ],
        closing_insight="Connect the idea to the reader's goal.",
        key_points_covered=[str(section["key_points"][0])],
    )


class ParallelWritingModel:
    def __init__(self) -> None:
        self.active_writers = 0
        self.max_active_writers = 0
        self.completion_order: list[int] = []

    async def generate_structured(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        if schema is ResearchDecision:
            return schema.model_validate(
                {
                    "selected_mode": "model_knowledge",
                    "rationale": "A general draft is sufficient.",
                    "estimated_knowledge_sufficient": True,
                    "confidence": 0.8,
                }
            )
        if schema is SEOPlan:
            return schema.model_validate(_plan())
        if schema is ArticleBookends:
            return schema.model_validate(
                {
                    "introduction": "Introduce the topic and what readers will learn.",
                    "conclusion": "Review the most important takeaways.",
                }
            )
        if schema is SectionDraft:
            section = json.loads(user_prompt)["section"]
            position = int(section["position"])
            self.active_writers += 1
            self.max_active_writers = max(
                self.max_active_writers, self.active_writers
            )
            await asyncio.sleep((2 - position) * 0.01)
            self.active_writers -= 1
            self.completion_order.append(position)
            return schema.model_validate(_draft(section))
        raise AssertionError(f"Unexpected requested schema: {schema.__name__}")


def test_workflow_writes_sections_concurrently_and_orders_by_plan() -> None:
    model = ParallelWritingModel()
    graph = build_blog_workflow_graph(model)

    result = asyncio.run(
        graph.ainvoke({"request": BlogRequest(topic="A useful topic")})
    )

    assert model.max_active_writers > 1
    assert model.completion_order != [0, 1, 2]
    assert [
        draft.position for draft in result["ordered_section_drafts"]
    ] == [0, 1, 2]
    assert [
        draft.task_id for draft in result["ordered_section_drafts"]
    ] == ["section_1", "section_2", "section_3"]


def test_section_writer_uses_planned_identity_when_model_returns_different_values() -> None:
    section = _plan().sections[0]

    class WrongIdentityModel:
        async def generate_structured(
            self,
            *,
            schema: type[SchemaT],
            system_prompt: str,
            user_prompt: str,
        ) -> SchemaT:
            payload = json.loads(user_prompt)
            result = _draft(payload["section"]).model_copy(
                update={"task_id": "different_section"}
            )
            return schema.model_validate(result)

    draft = asyncio.run(
        write_section(
            request=BlogRequest(topic="A useful topic"),
            plan=_plan(),
            section=section,
            model=WrongIdentityModel(),
        )
    )

    assert draft.task_id == section.id
    assert draft.position == section.position
    assert draft.heading == section.heading


def _generated_image() -> GeneratedImage:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(buffer, format="PNG")
    return GeneratedImage(
        model_name="owner/open-image-model",
        prompt="Editorial illustration",
        alt_text="An editorial illustration",
        image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


class FakeImageProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def generate_image(self, prompt: str, alt_text: str) -> GeneratedImage:
        self.calls += 1
        if self.fail:
            raise ImageProviderError("Private provider detail must not leak.")
        assert "A useful topic" in prompt
        assert alt_text == "Illustration for A practical topic guide"
        return _generated_image()


def test_image_generation_is_skipped_when_not_requested() -> None:
    model = ParallelWritingModel()
    image_provider = FakeImageProvider()
    graph = build_blog_workflow_graph(model, image_provider=image_provider)

    result = asyncio.run(
        graph.ainvoke({"request": BlogRequest(topic="A useful topic")})
    )

    assert result["image_result"].status == "skipped"
    assert result["workflow_stage"] == "article_ready"
    assert image_provider.calls == 0
    assert result["article"].image_result.status == "skipped"


def test_image_generation_completes_when_requested() -> None:
    model = ParallelWritingModel()
    image_provider = FakeImageProvider()
    graph = build_blog_workflow_graph(model, image_provider=image_provider)

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

    assert result["image_result"].status == "completed"
    assert result["image_result"].image.model_name == "owner/open-image-model"
    assert result["workflow_stage"] == "article_ready"
    assert image_provider.calls == 1
    assert result["article"].image_result.status == "completed"


def test_image_provider_failure_is_reported_without_losing_article_drafts() -> None:
    model = ParallelWritingModel()
    image_provider = FakeImageProvider(fail=True)
    graph = build_blog_workflow_graph(model, image_provider=image_provider)

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

    assert result["image_result"].status == "failed"
    assert "article draft is available" in result["image_result"].message
    assert "Private provider detail" not in result["image_result"].message
    assert result["workflow_stage"] == "article_ready"
    assert result["article"].image_result.status == "failed"
    assert len(result["ordered_section_drafts"]) == 3
    assert "## Introduction" in result["article"].markdown


def test_requested_image_without_provider_is_an_explicit_optional_failure() -> None:
    graph = build_blog_workflow_graph(ParallelWritingModel())

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

    assert result["image_result"].status == "failed"
    assert "no image provider is configured" in result["image_result"].message
    assert len(result["ordered_section_drafts"]) == 3
    assert result["article"].image_result.status == "failed"
