"""Shared state schema for the LangGraph workflow."""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from seo_blog_assistant.schemas import (
    ArticleBookends,
    ArticleResult,
    BlogRequest,
    ImageGenerationResult,
    ImagePlaceholder,
    InlineImage,
    PlannedSection,
    ResearchBundle,
    ResearchDecision,
    SEOPlan,
    SectionDraft,
)
from seo_blog_assistant.workflow.batched_writing import WrittenArticle
from seo_blog_assistant.workflow.topic_profiles import TopicCategory


class BlogWorkflowState(TypedDict):
    """Fields shared by the sequential and parallel workflow nodes."""

    request: BlogRequest
    topic_category: NotRequired[TopicCategory]
    content_angle: NotRequired[str | None]
    research_decision: NotRequired[ResearchDecision]
    research_bundle: NotRequired[ResearchBundle]
    seo_plan: NotRequired[SEOPlan]
    workflow_stage: NotRequired[
        Literal[
            "research_required",
            "research_started",
            "research_completed",
            "plan_ready",
            "images_skipped",
            "images_completed",
            "images_failed",
            "article_ready",
        ]
    ]
    active_section: NotRequired[PlannedSection]
    section_drafts: NotRequired[Annotated[list[SectionDraft], operator.add]]
    warnings: NotRequired[Annotated[list[str], operator.add]]
    ordered_section_drafts: NotRequired[list[SectionDraft]]
    image_result: NotRequired[ImageGenerationResult]
    inline_images: NotRequired[list[InlineImage]]
    image_placeholders: NotRequired[list[ImagePlaceholder]]
    written_article: NotRequired[WrittenArticle]
    batched_markdown: NotRequired[str]
    bookends: NotRequired[ArticleBookends]
    article: NotRequired[ArticleResult]
