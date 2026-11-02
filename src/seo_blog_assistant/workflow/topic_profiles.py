"""Compact category guidance shared by planning, research, and writing."""

from dataclasses import dataclass
from typing import Literal

TopicCategory = Literal[
    "technology",
    "sports",
    "food",
    "politics",
    "business_finance",
    "health_science",
    "travel_culture",
    "entertainment",
    "history_education",
    "general",
]


@dataclass(frozen=True)
class TopicProfile:
    """Editorial and evidence requirements for one topic category."""

    category: TopicCategory
    label: str
    research_policy: Literal["required", "recommended", "optional"]
    outline_elements: tuple[str, ...]
    evidence_rules: str
    style_notes: str
    search_hints: tuple[str, ...]
    evidence_items: int
    evidence_characters: int


TOPIC_PROFILES: dict[TopicCategory, TopicProfile] = {
    "technology": TopicProfile(
        category="technology",
        label="Technology",
        research_policy="recommended",
        outline_elements=(
            "problem or need",
            "how it works",
            "limitations",
            "adoption and real use cases",
            "solutions and best practices",
            "trade-offs",
        ),
        evidence_rules=(
            "Prefer sourced claims for versions, benchmarks, and release dates."
        ),
        style_notes=(
            "Explain technical ideas clearly and distinguish capabilities from "
            "trade-offs."
        ),
        search_hints=(),
        evidence_items=4,
        evidence_characters=600,
    ),
    "sports": TopicProfile(
        category="sports",
        label="Sports",
        research_policy="required",
        outline_elements=(
            "current context, recent results, and standings",
            "key statistics and records",
            "historical comparisons, past matches, or seasons",
            "key players or teams and turning points",
            "analysis of why events happened",
            "what to watch next",
        ),
        evidence_rules=(
            "Every score, statistic, record, date, and ranking must come from supplied "
            "research with its citation label. If a figure is missing, say it is "
            "unavailable; never estimate or invent statistics."
        ),
        style_notes=(
            "Separate sourced match facts from analysis and clearly identify the "
            "relevant season."
        ),
        search_hints=("latest results stats", "history records past seasons"),
        evidence_items=6,
        evidence_characters=600,
    ),
    "food": TopicProfile(
        category="food",
        label="Food",
        research_policy="optional",
        outline_elements=(
            "origin and cultural background",
            "ingredients and flavor",
            "preparation methods and technique",
            "regional variations",
            "nutrition, health, food-safety, or allergen notes",
            "serving, pairing, and common mistakes",
        ),
        evidence_rules=(
            "Cite nutrition facts and health claims when research is supplied; "
            "quantities and techniques may use general knowledge."
        ),
        style_notes=(
            "Use sensory, practical language and respect the dish's cultural "
            "context."
        ),
        search_hints=(),
        evidence_items=3,
        evidence_characters=500,
    ),
    "politics": TopicProfile(
        category="politics",
        label="Politics and democracy",
        research_policy="required",
        outline_elements=(
            "background and history",
            "key facts and timeline",
            "institutions and rules",
            "fair presentation of competing perspectives",
            "sourced data, results, polls, or laws",
            "impact, consequences, and what to watch",
        ),
        evidence_rules=(
            "Attribute every claim to a cited source, separate fact from opinion, and "
            "present main competing positions fairly. Never endorse or attack a party "
            "or candidate, or invent quotes, polls, or results."
        ),
        style_notes=(
            "Use neutral language, distinguish verified facts from interpretation, "
            "and represent viewpoints fairly."
        ),
        search_hints=("facts timeline", "results analysis"),
        evidence_items=6,
        evidence_characters=600,
    ),
    "business_finance": TopicProfile(
        category="business_finance",
        label="Business and finance",
        research_policy="required",
        outline_elements=(
            "context and market background",
            "key numbers and trends",
            "relevant case examples",
            "risks and limitations",
            "outlook",
        ),
        evidence_rules=(
            "Cite all figures and include a one-line note that the article is not "
            "financial advice."
        ),
        style_notes=(
            "Explain business implications plainly and distinguish evidence from "
            "forecasts."
        ),
        search_hints=("latest figures", "analysis outlook"),
        evidence_items=6,
        evidence_characters=600,
    ),
    "health_science": TopicProfile(
        category="health_science",
        label="Health and science",
        research_policy="recommended",
        outline_elements=(
            "what it is",
            "what research shows",
            "benefits and risks",
            "who is affected",
            "practical takeaways",
            "when to consult a professional",
        ),
        evidence_rules=(
            "Cite studies or health bodies for claims; do not provide dosing "
            "instructions and include a one-line note that the article is not "
            "medical advice."
        ),
        style_notes=(
            "Use cautious, accessible language and distinguish evidence strength "
            "from established fact."
        ),
        search_hints=("research evidence",),
        evidence_items=4,
        evidence_characters=600,
    ),
    "travel_culture": TopicProfile(
        category="travel_culture",
        label="Travel and culture",
        research_policy="optional",
        outline_elements=(
            "overview and why it matters",
            "history and culture",
            "what to see or do",
            "practical tips on seasons, costs, and transport",
            "local etiquette and safety",
        ),
        evidence_rules=(
            'Cite prices, opening hours, and rules when research is supplied; '
            'otherwise mark them "check locally".'
        ),
        style_notes=(
            "Balance vivid cultural context with practical, respectful travel "
            "guidance."
        ),
        search_hints=(),
        evidence_items=3,
        evidence_characters=500,
    ),
    "entertainment": TopicProfile(
        category="entertainment",
        label="Entertainment",
        research_policy="recommended",
        outline_elements=(
            "background",
            "key works or people",
            "sourced reception, sales, ratings, or awards",
            "cultural impact",
            "what comes next",
        ),
        evidence_rules="Cite ratings, dates, sales, and awards.",
        style_notes=(
            "Separate personal or critical interpretation from sourced reception "
            "and performance."
        ),
        search_hints=("reviews ratings",),
        evidence_items=4,
        evidence_characters=600,
    ),
    "history_education": TopicProfile(
        category="history_education",
        label="History and education",
        research_policy="optional",
        outline_elements=(
            "context and timeline",
            "key events and people",
            "causes and effects",
            "different interpretations",
            "lessons or relevance today",
        ),
        evidence_rules=(
            "Cite dates and quotations when research is supplied; do not invent "
            "quotations."
        ),
        style_notes=(
            "Distinguish established evidence from interpretation and acknowledge "
            "meaningful disagreement."
        ),
        search_hints=(),
        evidence_items=3,
        evidence_characters=500,
    ),
    "general": TopicProfile(
        category="general",
        label="General",
        research_policy="optional",
        outline_elements=(
            "reader need and context",
            "core ideas",
            "practical examples or applications",
            "challenges and useful considerations",
            "practical guidance",
            "conclusion and next steps",
        ),
        evidence_rules="Do not invent statistics or sources.",
        style_notes="Use a clear, balanced structure that fits the audience and topic.",
        search_hints=(),
        evidence_items=3,
        evidence_characters=500,
    ),
}


def get_profile(category: str | None) -> TopicProfile:
    """Return a known category profile or the general fallback."""
    for profile in TOPIC_PROFILES.values():
        if category == profile.category:
            return profile
    return TOPIC_PROFILES["general"]


def profile_prompt_block(profile: TopicProfile) -> str:
    """Render only the selected profile as a compact prompt block."""
    outline = "; ".join(profile.outline_elements)
    return (
        f"Category: {profile.label} ({profile.category}). "
        f"Outline options: {outline}. "
        f"Evidence: {profile.evidence_rules} "
        f"Style: {profile.style_notes}"
    )
