import asyncio
import base64
import json
from io import BytesIO
from typing import TypeVar

import pytest
from pydantic import BaseModel
from PIL import Image

from seo_blog_assistant.config import Settings
from seo_blog_assistant.providers.image import ImageProviderError
from seo_blog_assistant.schemas import (
    ArticleBookends,
    BlogRequest,
    DraftSubsection,
    GeneratedImage,
    ResearchDecision,
    SEOPlan,
    SectionDraft,
)
from seo_blog_assistant.providers.text_model import TextResult
from seo_blog_assistant.workflow.graph import build_blog_workflow_graph
from seo_blog_assistant.workflow.topic_profiles import get_profile
from seo_blog_assistant.workflow.writing import write_section

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOnlyFake:
    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> TextResult:
        raise AssertionError("This test fake only supports structured generation.")


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


class ParallelWritingModel(StructuredOnlyFake):
    def __init__(
        self,
        *,
        warn_on_first_section: bool = False,
        section_started: asyncio.Event | None = None,
        image_started: asyncio.Event | None = None,
        image_prompts: dict[int, str] | None = None,
    ) -> None:
        self.active_writers = 0
        self.max_active_writers = 0
        self.completion_order: list[int] = []
        self.warn_on_first_section = warn_on_first_section
        self.section_started = section_started
        self.image_started = image_started
        self.image_prompts = image_prompts or {}
        self.all_sections_started = asyncio.Event()
        self.section_finished = [asyncio.Event() for _ in range(3)]

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
            if self.active_writers == 3:
                self.all_sections_started.set()
            await self.all_sections_started.wait()
            if self.section_started is not None and self.image_started is not None:
                self.section_started.set()
                await self.image_started.wait()
            if position < 2:
                await self.section_finished[position + 1].wait()
            self.active_writers -= 1
            self.completion_order.append(position)
            self.section_finished[position].set()
            draft = _draft(section)
            if position in self.image_prompts:
                draft = draft.model_copy(
                    update={"image_prompt": self.image_prompts[position]}
                )
            if self.warn_on_first_section and position == 0:
                draft = draft.model_copy(
                    update={
                        "subsections": [
                            DraftSubsection(
                                heading="Incorrect heading",
                                content="Explain the key process.",
                            )
                        ]
                    }
                )
            return schema.model_validate(draft)
        raise AssertionError(f"Unexpected requested schema: {schema.__name__}")


def test_workflow_writes_sections_concurrently_and_orders_by_plan() -> None:
    model = ParallelWritingModel()
    graph = build_blog_workflow_graph(model, writing_mode="per_section")

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


def test_section_prompt_has_selected_profile_and_no_invented_figures_rule() -> None:
    class CapturingModel:
        calls: list[tuple[str, str]] = []

        async def generate_structured(
            self,
            *,
            schema: type[SchemaT],
            system_prompt: str,
            user_prompt: str,
        ) -> SchemaT:
            self.calls.append((system_prompt, user_prompt))
            payload = json.loads(user_prompt)
            return schema.model_validate(_draft(payload["section"]))

        async def generate_text(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            max_tokens: int,
        ) -> TextResult:
            raise AssertionError("Section writing uses structured generation.")

    model = CapturingModel()
    plan = _plan()

    asyncio.run(
        write_section(
            request=BlogRequest(topic="A sports topic"),
            plan=plan,
            section=plan.sections[0],
            model=model,
            profile=get_profile("sports"),
        )
    )

    system_prompt, user_prompt = model.calls[0]
    assert "Category: Sports" in user_prompt
    assert "Category: Food" not in user_prompt
    assert "Every score, statistic, record, date, and ranking" in user_prompt
    assert "Never guess statistics, dates, scores, records, polls" in system_prompt


def test_per_section_placeholder_prompts_follow_plan_order_without_provider_calls() -> None:
    prompt = (
        "A welcoming gardener tending a bright community garden at sunrise, "
        "editorial illustration, warm gentle colors, hopeful mood, wide balanced "
        "composition, layered plants, soft light, clear foreground, natural detail."
    )
    model = ParallelWritingModel(image_prompts={0: prompt, 1: prompt})

    class FailIfCalledImageProvider:
        calls = 0

        async def generate_image(self, _prompt: str, _alt_text: str) -> GeneratedImage:
            self.calls += 1
            raise AssertionError("placeholder mode must not call image inference")

    provider = FailIfCalledImageProvider()
    graph = build_blog_workflow_graph(
        model,
        image_provider=provider,
        writing_mode="per_section",
        max_images=2,
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

    placeholders = result["article"].image_placeholders
    assert [item.image_id for item in placeholders] == ["img_1", "img_2"]
    assert [item.after_heading for item in placeholders] == [
        "Section 1",
        "Section 2",
    ]
    assert all(item.prompt == prompt for item in placeholders)
    assert provider.calls == 0
    assert result["article"].image_result.status == "skipped"
    assert result["article"].inline_images == []


def test_workflow_forwards_section_repair_warnings_to_article_result() -> None:
    model = ParallelWritingModel(warn_on_first_section=True)
    graph = build_blog_workflow_graph(model, writing_mode="per_section")

    result = asyncio.run(
        graph.ainvoke({"request": BlogRequest(topic="A useful topic")})
    )

    assert result["article"].warnings
    assert any("corrected subsection headings" in warning for warning in result["article"].warnings)


def test_section_writer_uses_planned_identity_when_model_returns_different_values() -> None:
    section = _plan().sections[0]

    class WrongIdentityModel(StructuredOnlyFake):
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

    draft, warnings = asyncio.run(
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
    assert warnings == []


def test_section_writer_retries_then_repairs_headings_and_unknown_sources() -> None:
    section = _plan().sections[0]

    class ImperfectModel(StructuredOnlyFake):
        def __init__(self) -> None:
            self.prompts: list[dict[str, object]] = []

        async def generate_structured(
            self,
            *,
            schema: type[SchemaT],
            system_prompt: str,
            user_prompt: str,
        ) -> SchemaT:
            payload = json.loads(user_prompt)
            self.prompts.append(payload)
            response = _draft(payload["section"]).model_copy(
                update={
                    "subsections": [
                        DraftSubsection(
                            heading="Wrong heading",
                            content="A statement with an invented [source_missing].",
                            source_ids=["source_missing"],
                        )
                    ],
                    "key_points_covered": ["  EXPLAIN   KEY IDEA 1  "],
                }
            )
            return schema.model_validate(response)

    model = ImperfectModel()
    draft, warnings = asyncio.run(
        write_section(
            request=BlogRequest(topic="A useful topic"),
            plan=_plan(),
            section=section,
            model=model,
        )
    )

    assert len(model.prompts) == 2
    assert model.prompts[1]["validation_feedback"]
    assert draft.subsections[0].heading == section.h3_headings[0]
    assert draft.subsections[0].source_ids == []
    assert any("corrected subsection headings" in warning for warning in warnings)
    assert any("removed citations" in warning for warning in warnings)
    assert not any("omitted planned key points" in warning for warning in warnings)


def test_section_writer_warns_when_key_points_remain_uncovered() -> None:
    section = _plan().sections[0]

    class MissingKeyPointModel(StructuredOnlyFake):
        async def generate_structured(
            self,
            *,
            schema: type[SchemaT],
            system_prompt: str,
            user_prompt: str,
        ) -> SchemaT:
            payload = json.loads(user_prompt)
            response = _draft(payload["section"]).model_copy(
                update={"key_points_covered": ["Unrelated point"]}
            )
            return schema.model_validate(response)

    draft, warnings = asyncio.run(
        write_section(
            request=BlogRequest(topic="A useful topic"),
            plan=_plan(),
            section=section,
            model=MissingKeyPointModel(),
        )
    )

    assert draft.key_points_covered == ["Unrelated point"]
    assert any("omitted planned key points" in warning for warning in warnings)


def test_section_writer_still_rejects_wrong_subsection_count() -> None:
    section = _plan().sections[0]

    class WrongCountModel(StructuredOnlyFake):
        async def generate_structured(
            self,
            *,
            schema: type[SchemaT],
            system_prompt: str,
            user_prompt: str,
        ) -> SchemaT:
            payload = json.loads(user_prompt)
            response = _draft(payload["section"]).model_copy(
                update={
                    "subsections": [
                        DraftSubsection(heading="First", content="First."),
                        DraftSubsection(heading="Second", content="Second."),
                    ]
                }
            )
            return schema.model_validate(response)

    with pytest.raises(ValueError, match="wrong number of subsections"):
        asyncio.run(
            write_section(
                request=BlogRequest(topic="A useful topic"),
                plan=_plan(),
                section=section,
                model=WrongCountModel(),
            )
        )


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
            raise ImageProviderError(
                "Image generation failed (model=test, provider=auto): rate limited"
            )
        assert "A useful topic" in prompt
        assert alt_text == "Illustration for A practical topic guide"
        return _generated_image()


def test_images_run_concurrently_with_section_writing() -> None:
    section_started = asyncio.Event()
    image_started = asyncio.Event()
    model = ParallelWritingModel(
        section_started=section_started,
        image_started=image_started,
    )

    class BarrierImageProvider:
        async def generate_image(self, _prompt: str, _alt_text: str) -> GeneratedImage:
            image_started.set()
            await section_started.wait()
            return _generated_image()

    graph = build_blog_workflow_graph(
        model,
        image_provider=BarrierImageProvider(),
        writing_mode="per_section",
        settings=Settings(_env_file=None, image_mode="generate"),
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

    assert result["image_result"].status == "completed"
    assert result["article"].image_result.status == "completed"


def test_image_generation_is_skipped_when_not_requested() -> None:
    model = ParallelWritingModel()
    image_provider = FakeImageProvider()
    graph = build_blog_workflow_graph(
        model,
        image_provider=image_provider,
        writing_mode="per_section",
        settings=Settings(_env_file=None, image_mode="generate"),
    )

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
    graph = build_blog_workflow_graph(
        model,
        image_provider=image_provider,
        writing_mode="per_section",
        settings=Settings(_env_file=None, image_mode="generate"),
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

    assert result["image_result"].status == "completed"
    assert result["image_result"].image.model_name == "owner/open-image-model"
    assert result["workflow_stage"] == "article_ready"
    assert image_provider.calls == 1
    assert result["article"].image_result.status == "completed"


def test_image_generation_respects_zero_image_limit() -> None:
    model = ParallelWritingModel()
    image_provider = FakeImageProvider()
    graph = build_blog_workflow_graph(
        model,
        image_provider=image_provider,
        writing_mode="per_section",
        max_images=0,
        settings=Settings(_env_file=None, image_mode="generate"),
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

    assert result["image_result"].status == "skipped"
    assert "image limit" in result["image_result"].message
    assert image_provider.calls == 0


def test_image_provider_failure_is_reported_without_losing_article_drafts() -> None:
    model = ParallelWritingModel()
    image_provider = FakeImageProvider(fail=True)
    graph = build_blog_workflow_graph(
        model,
        image_provider=image_provider,
        writing_mode="per_section",
        settings=Settings(_env_file=None, image_mode="generate"),
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

    assert result["image_result"].status == "failed"
    assert "rate limited" in result["image_result"].message
    assert result["workflow_stage"] == "article_ready"
    assert result["article"].image_result.status == "failed"
    assert len(result["ordered_section_drafts"]) == 3
    assert "## Introduction" in result["article"].markdown


def test_unexpected_image_provider_error_does_not_fail_article() -> None:
    class UnexpectedImageProvider:
        async def generate_image(self, _prompt: str, _alt_text: str) -> GeneratedImage:
            raise ValueError("private provider details")

    graph = build_blog_workflow_graph(
        ParallelWritingModel(),
        image_provider=UnexpectedImageProvider(),
        writing_mode="per_section",
        settings=Settings(_env_file=None, image_mode="generate"),
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

    assert result["image_result"].status == "failed"
    assert "private provider details" not in result["image_result"].message
    assert result["article"].title == "A practical topic guide"


def test_requested_image_without_provider_is_an_explicit_optional_failure() -> None:
    graph = build_blog_workflow_graph(
        ParallelWritingModel(),
        writing_mode="per_section",
        settings=Settings(_env_file=None, image_mode="generate"),
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

    assert result["image_result"].status == "failed"
    assert "no image provider is configured" in result["image_result"].message
    assert len(result["ordered_section_drafts"]) == 3
    assert result["article"].image_result.status == "failed"
