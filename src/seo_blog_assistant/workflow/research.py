"""Research-mode assessment and deterministic workflow routing."""

import json

from seo_blog_assistant.providers.text_model import StructuredTextModel
from seo_blog_assistant.schemas import BlogRequest, ResearchDecision, ResearchMode
from seo_blog_assistant.workflow.topic_profiles import get_profile

RESEARCH_ASSESSMENT_SYSTEM_PROMPT = """\
You assess whether a blog topic can reasonably be drafted from a language \
model's general knowledge or needs web research.

Choose web_research when current information, verification, or source citations \
are important. Otherwise, estimate whether model_knowledge is sufficient for \
a useful general draft.

Classify the topic as one of: technology (software and engineering), sports \
(sport and competition), food (cooking and cuisine), politics (government and \
democracy), business_finance (companies and markets), health_science (medicine \
and research), travel_culture (places and customs), entertainment (media and \
arts), history_education (past events and learning), general (other topics). \
Return topic_category and a concise content_angle, such as "season preview with \
historical comparison".

This is only an estimate, not proof that facts are correct or current. Do not \
claim that a web search or fact verification has already happened. Return the \
requested structured result with an honest rationale and confidence.
"""


async def assess_research(
    request: BlogRequest,
    model: StructuredTextModel,
    *,
    search_provider_configured: bool = True,
) -> ResearchDecision:
    """Ask the model to assess research needs and enforce explicit requirements."""
    assessment_fields = {
        "topic",
        "audience",
        "target_country",
        "category",
        "seo_goals",
        "require_freshness",
        "require_citations",
    }
    request_payload = request.model_dump(
        mode="json",
        exclude_none=True,
        include=assessment_fields,
    )
    request_payload = {
        key: value
        for key, value in request_payload.items()
        if not (isinstance(value, list) and not value)
    }
    user_prompt = json.dumps(request_payload, ensure_ascii=False)
    decision = await model.generate_structured(
        schema=ResearchDecision,
        system_prompt=RESEARCH_ASSESSMENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    decision = ResearchDecision.model_validate(decision)

    category = request.category or decision.topic_category
    profile = get_profile(category)
    rationale = decision.rationale
    selected_mode = decision.selected_mode
    if profile.research_policy == "required":
        if selected_mode != "web_research":
            rationale += (
                f" Web research is required for {profile.label.lower()} topics "
                "to verify important facts and figures."
            )
        selected_mode = "web_research"
    elif profile.research_policy == "recommended":
        if search_provider_configured:
            if selected_mode != "web_research":
                rationale += (
                    f" Web research is recommended for {profile.label.lower()} "
                    "topics when a search provider is available."
                )
            selected_mode = "web_research"
        else:
            selected_mode = "model_knowledge"
            rationale += (
                " Research was recommended for this topic, but no search "
                "provider is configured."
            )

    angle = (
        decision.content_angle
        if request.category is None
        else f"{profile.label} article about {request.topic}"
    )
    decision = decision.model_copy(
        update={
            "topic_category": category,
            "content_angle": angle,
            "selected_mode": selected_mode,
            "rationale": rationale,
        }
    )

    required_reasons = []
    if request.require_freshness:
        required_reasons.append("The user requires fresh information.")
    if request.require_citations:
        required_reasons.append("The user requires source citations.")
    if required_reasons and decision.selected_mode != "web_research":
        rationale = (
            f"{decision.rationale} "
            f"Web research was selected because {' '.join(required_reasons)}"
        )
        return decision.model_copy(
            update={"selected_mode": "web_research", "rationale": rationale}
        )
    return decision


def research_route(decision: ResearchDecision) -> ResearchMode:
    """Return the chosen graph route without claiming a search has happened."""
    return decision.selected_mode


def should_search_web(decision: ResearchDecision) -> bool:
    """Whether the workflow should enter the web-research branch."""
    return research_route(decision) == "web_research"
