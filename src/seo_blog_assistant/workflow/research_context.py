"""Compact retrieved evidence before including it in model prompts."""

from typing import TypedDict

from seo_blog_assistant.schemas import ResearchBundle

MAX_PROMPT_EVIDENCE_ITEMS = 3
MAX_PROMPT_EVIDENCE_CHARACTERS = 1000


class PromptEvidence(TypedDict):
    evidence_id: str
    source_id: str
    text: str


class PromptResearchContext(TypedDict):
    provider: str
    search_queries: list[str]
    sources: list[dict[str, str | None]]
    evidence: list[PromptEvidence]


def compact_research_context(
    research: ResearchBundle | None,
) -> PromptResearchContext | None:
    """Keep provenance while bounding evidence included in each model call."""
    if research is None:
        return None

    evidence = research.evidence[:MAX_PROMPT_EVIDENCE_ITEMS]
    cited_source_ids = {item.source_id for item in evidence}
    sources = [
        {
            "source_id": source.source_id,
            "url": str(source.url),
            "title": source.title,
            "publisher": source.publisher,
            "published_at": source.published_at,
        }
        for source in research.sources
        if source.source_id in cited_source_ids
    ]
    return {
        "provider": research.provider,
        "search_queries": research.search_queries[:3],
        "sources": sources,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "source_id": item.source_id,
                "text": item.text[:MAX_PROMPT_EVIDENCE_CHARACTERS],
            }
            for item in evidence
        ],
    }
