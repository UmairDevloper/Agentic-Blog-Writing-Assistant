"""Select and compact retrieved evidence for model prompts."""

import re
from typing import TypedDict

from seo_blog_assistant.schemas import EvidenceSnippet, ResearchBundle
from seo_blog_assistant.workflow.topic_profiles import TopicProfile

MAX_PROMPT_EVIDENCE_ITEMS = 4
MAX_PROMPT_EVIDENCE_CHARACTERS = 600

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


class PromptEvidence(TypedDict):
    evidence_id: str
    source_id: str
    text: str


class PromptSource(TypedDict, total=False):
    source_id: str
    label: str
    title: str


class PromptResearchContext(TypedDict):
    provider: str
    sources: list[PromptSource]
    evidence: list[PromptEvidence]


def compact_research_context(
    research: ResearchBundle | None,
    query_terms: list[str] | None = None,
    *,
    labels: bool = False,
    profile: TopicProfile | None = None,
) -> PromptResearchContext | None:
    """Keep provenance while selecting bounded and relevant evidence."""
    if research is None:
        return None

    evidence_items = profile.evidence_items if profile else MAX_PROMPT_EVIDENCE_ITEMS
    evidence_characters = (
        profile.evidence_characters
        if profile
        else MAX_PROMPT_EVIDENCE_CHARACTERS
    )
    round_robin = profile is not None and profile.category in {"sports", "politics"}
    evidence = _select_evidence(
        research,
        query_terms,
        limit=evidence_items,
        round_robin=round_robin,
    )
    label_map = (
        build_research_labels(
            research,
            query_terms,
            profile=profile,
        )
        if labels
        else {}
    )
    source_labels = {source_id: label for label, source_id in label_map.items()}
    cited_source_ids = {item.source_id for item in evidence}
    sources = [
        (
            {"label": source_labels[source.source_id], "title": source.title[:160]}
            if labels
            else {"source_id": source.source_id, "title": source.title[:160]}
        )
        for source in research.sources
        if source.source_id in cited_source_ids
    ]
    return {
        "provider": research.provider,
        "sources": sources,
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "source_id": source_labels.get(item.source_id, item.source_id),
                "text": item.text[:evidence_characters],
            }
            for item in evidence
        ],
    }


def build_research_labels(
    research: ResearchBundle | None,
    query_terms: list[str] | None = None,
    *,
    profile: TopicProfile | None = None,
) -> dict[str, str]:
    """Map compact-context source labels to original research source IDs."""
    if research is None:
        return {}
    limit = profile.evidence_items if profile else MAX_PROMPT_EVIDENCE_ITEMS
    round_robin = profile is not None and profile.category in {"sports", "politics"}
    selected_source_ids = {
        item.source_id
        for item in _select_evidence(
            research,
            query_terms,
            limit=limit,
            round_robin=round_robin,
        )
    }
    return {
        f"S{index}": source.source_id
        for index, source in enumerate(
            (
                source
                for source in research.sources
                if source.source_id in selected_source_ids
            ),
            start=1,
        )
    }


def _select_evidence(
    research: ResearchBundle,
    query_terms: list[str] | None,
    *,
    limit: int,
    round_robin: bool,
) -> list[EvidenceSnippet]:
    if query_terms is not None and not round_robin:
        query_words = {
            word.casefold()
            for term in query_terms
            for word in _WORD_PATTERN.findall(term)
        }
        ranked = sorted(
            enumerate(research.evidence),
            key=lambda entry: (
                -len(
                    query_words
                    & {
                        word.casefold()
                        for word in _WORD_PATTERN.findall(entry[1].text)
                    }
                ),
                entry[0],
            ),
        )
        return [
            evidence
            for _, evidence in             ranked[:limit]
        ]

    evidence_by_source = {
        source.source_id: []
        for source in research.sources
    }
    for evidence in research.evidence:
        evidence_by_source[evidence.source_id].append(evidence)
    if query_terms is not None:
        query_words = {
            word.casefold()
            for term in query_terms
            for word in _WORD_PATTERN.findall(term)
        }
        for source_evidence in evidence_by_source.values():
            source_evidence.sort(
                key=lambda item: -len(
                    query_words
                    & {
                        word.casefold()
                        for word in _WORD_PATTERN.findall(item.text)
                    }
                )
            )

    selected: list[EvidenceSnippet] = []
    round_index = 0
    while len(selected) < limit:
        added_in_round = False
        for source in research.sources:
            source_evidence = evidence_by_source[source.source_id]
            if round_index < len(source_evidence):
                selected.append(source_evidence[round_index])
                added_in_round = True
                if len(selected) == limit:
                    break
        if not added_in_round:
            break
        round_index += 1
    return selected
