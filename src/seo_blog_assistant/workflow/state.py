"""Shared state schema for the LangGraph workflow."""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from seo_blog_assistant.schemas import (
    ArticleBookends,
    ArticleResult,
    BlogRequest,
    ImageGenerationResult,
    PlannedSection,
    ResearchBundle,
    ResearchDecision,
    SEOPlan,
    SectionDraft,
)


class BlogWorkflowState(TypedDict):
    """Fields shared by the sequential and parallel workflow nodes."""

    request: BlogRequest
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
    ordered_section_drafts: NotRequired[list[SectionDraft]]
    image_result: NotRequired[ImageGenerationResult]
    bookends: NotRequired[ArticleBookends]
    article: NotRequired[ArticleResult]
