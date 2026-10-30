"""Research-mode assessment and deterministic workflow routing."""

import json

from seo_blog_assistant.providers.text_model import StructuredTextModel
from seo_blog_assistant.schemas import BlogRequest, ResearchDecision, ResearchMode

RESEARCH_ASSESSMENT_SYSTEM_PROMPT = """\
You assess whether a blog topic can reasonably be drafted from a language \
model's general knowledge or needs web research.

Choose web_research when current information, verification, or source citations \
are important. Otherwise, estimate whether model_knowledge is sufficient for \
a useful general draft.

This is only an estimate, not proof that facts are correct or current. Do not \
claim that a web search or fact verification has already happened. Return the \
requested structured result with an honest rationale and confidence.
"""


async def assess_research(
    request: BlogRequest,
    model: StructuredTextModel,
) -> ResearchDecision:
    """Ask the model to assess research needs and enforce explicit requirements."""
    user_prompt = json.dumps(
        {
            "topic": request.topic,
            "audience": request.audience,
            "target_country": request.target_country,
            "seo_goals": request.seo_goals,
            "require_freshness": request.require_freshness,
            "require_citations": request.require_citations,
        },
        ensure_ascii=False,
    )
    decision = await model.generate_structured(
        schema=ResearchDecision,
        system_prompt=RESEARCH_ASSESSMENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    decision = ResearchDecision.model_validate(decision)

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
