"""SEO-plan generation using the shared structured model interface."""

import json
import logging

from seo_blog_assistant.providers.text_model import StructuredTextModel
from seo_blog_assistant.schemas import (
    BlogRequest,
    ResearchBundle,
    ResearchDecision,
    SEOPlan,
)
from seo_blog_assistant.workflow.research_context import compact_research_context
from seo_blog_assistant.workflow.topic_profiles import (
    get_profile,
    profile_prompt_block,
)

logger = logging.getLogger(__name__)

SEO_PLAN_SYSTEM_PROMPT = """\
Create an ordered SEO outline and brief, not finished article prose. Keep each \
section purpose to one sentence, use at most 4 key points and 2-3 H3 headings \
per section, and return at most 8 related keyword ideas. Make budgets total \
within 10% (or 50 words) of the requested length. Use supplied research only; \
never invent sources or claim verification. Return only the structured plan.
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

    profile = get_profile(decision.topic_category)
    request_payload = request.model_dump(mode="json", exclude_none=True)
    request_payload = {
        key: value
        for key, value in request_payload.items()
        if not (isinstance(value, list) and not value)
    }
    prompt_payload: dict[str, object] = {
        "request": request_payload,
        "research_mode": decision.selected_mode,
        "research_rationale": decision.rationale,
        "topic_profile": profile_prompt_block(profile),
    }
    if decision.content_angle:
        prompt_payload["content_angle"] = decision.content_angle
    research_context = compact_research_context(research, profile=profile)
    if research_context is not None:
        prompt_payload["research_bundle"] = research_context
    user_prompt = json.dumps(prompt_payload, ensure_ascii=False)
    system_lines = [
        SEO_PLAN_SYSTEM_PROMPT,
        "Shape the outline from the selected profile's outline elements, choosing "
        "only what fits and dropping the rest rather than padding. Follow the "
        "category evidence rules. Do not force problem, limitations, or solutions "
        "sections on non-technical topics.",
    ]
    if profile.category in {"sports", "politics"}:
        system_lines.append(
            "Include at least one history or background section; when research is "
            "available, include a section built around key numbers or data."
        )
    plan = await model.generate_structured(
        schema=SEOPlan,
        system_prompt="\n".join(system_lines),
        user_prompt=user_prompt,
    )
    plan = SEOPlan.model_validate(plan)
    if plan.target_words != request.target_words:
        plan_data = plan.model_dump()
        plan_data["target_words"] = request.target_words
        plan = SEOPlan.model_validate(plan_data)
        logger.warning(
            "Repaired SEO plan target_words to match requested target=%s",
            request.target_words,
        )
    return plan
