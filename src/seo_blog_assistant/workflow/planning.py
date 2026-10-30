"""SEO-plan generation using the shared structured model interface."""

import json

from seo_blog_assistant.providers.text_model import StructuredTextModel
from seo_blog_assistant.schemas import (
    BlogRequest,
    ResearchBundle,
    ResearchDecision,
    SEOPlan,
)
from seo_blog_assistant.workflow.research_context import compact_research_context

SEO_PLAN_SYSTEM_PROMPT = """\
You are an SEO content planner. Create a structured outline and planning \
briefs, not finished article prose. The introduction and conclusion fields \
must be instructions for a later writer, not paragraphs for the reader.

Make the outline useful for the requested audience, tone, country, length, and \
SEO goals. Give every section a distinct purpose, concrete key points, exact \
H3 headings, and a realistic word budget. Return sections in intended article \
order; section positions are zero-based and must match their list order. \
Include introduction, section, and \
conclusion word budgets whose total is within 10% (or 50 words) of the \
requested article length.

Use supplied research evidence when available. Never invent sources, \
citations, or claims of verification. If no research bundle is supplied, do \
not imply that a search or verification occurred. Return only the requested \
structured plan.
"""


async def create_seo_plan(
    request: BlogRequest,
    decision: ResearchDecision,
    model: StructuredTextModel,
    research: ResearchBundle | None = None,
) -> SEOPlan:
    """Generate an ordered SEO plan and require evidence for research mode."""
    if decision.selected_mode == "web_research" and research is None:
        raise ValueError(
            "a completed ResearchBundle is required before planning in web_research mode"
        )
    if decision.selected_mode == "model_knowledge" and research is not None:
        raise ValueError(
            "research evidence cannot be attached when model_knowledge mode was selected"
        )

    user_prompt = json.dumps(
        {
            "request": request.model_dump(mode="json"),
            "research_mode": decision.selected_mode,
            "research_rationale": decision.rationale,
            "research_bundle": compact_research_context(research),
        },
        ensure_ascii=False,
    )
    plan = await model.generate_structured(
        schema=SEOPlan,
        system_prompt=SEO_PLAN_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    plan = SEOPlan.model_validate(plan)
    if plan.target_words != request.target_words:
        raise ValueError(
            "SEO plan target_words must match the requested article length"
        )
    return plan
