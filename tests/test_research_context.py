from seo_blog_assistant.schemas import ResearchBundle
from seo_blog_assistant.workflow.research_context import (
    MAX_PROMPT_EVIDENCE_CHARACTERS,
    MAX_PROMPT_EVIDENCE_ITEMS,
    build_research_labels,
    compact_research_context,
)
from seo_blog_assistant.workflow.topic_profiles import get_profile


def test_compact_research_context_is_bounded_and_omits_urls_and_queries() -> None:
    bundle = ResearchBundle(
        search_performed=True,
        provider="tavily",
        searched_at="2026-10-30T12:00:00Z",
        search_queries=["topic evidence"],
        sources=[
            {
                "source_id": f"source_{position}",
                "url": f"https://example.com/{position}",
                "title": f"Guide {position}",
            }
            for position in range(5)
        ],
        evidence=[
            {
                "evidence_id": f"evidence_{position}",
                "source_id": f"source_{position}",
                "text": "x" * (MAX_PROMPT_EVIDENCE_CHARACTERS * 2),
            }
            for position in range(5)
        ],
    )

    compact = compact_research_context(bundle)

    assert compact is not None
    assert len(compact["evidence"]) == MAX_PROMPT_EVIDENCE_ITEMS
    assert len(compact["sources"]) == MAX_PROMPT_EVIDENCE_ITEMS
    assert all(
        len(item["text"]) == MAX_PROMPT_EVIDENCE_CHARACTERS
        for item in compact["evidence"]
    )
    assert "search_queries" not in compact
    assert all("url" not in source and "publisher" not in source for source in compact["sources"])
    assert {item["source_id"] for item in compact["evidence"]} == {
        source["source_id"] for source in compact["sources"]
    }


def test_compact_research_context_is_none_without_research() -> None:
    assert compact_research_context(None) is None


def _bundle_with_evidence(
    evidence: list[tuple[str, str]],
    source_ids: list[str],
) -> ResearchBundle:
    return ResearchBundle(
        search_performed=True,
        provider="tavily",
        searched_at="2026-10-30T12:00:00Z",
        search_queries=["topic evidence"],
        sources=[
            {
                "source_id": source_id,
                "url": f"https://example.com/{source_id}",
                "title": f"Guide {source_id}",
            }
            for source_id in source_ids
        ],
        evidence=[
            {
                "evidence_id": f"evidence_{index}",
                "source_id": source_id,
                "text": text,
            }
            for index, (source_id, text) in enumerate(evidence)
        ],
    )


def test_research_context_round_robins_sources_without_query_terms() -> None:
    bundle = _bundle_with_evidence(
        [
            ("source_1", "First source evidence."),
            ("source_1", "Second source evidence."),
            ("source_1", "Third source evidence."),
            ("source_2", "Evidence from the second source."),
            ("source_2", "More from the second source."),
            ("source_3", "Evidence from the third source."),
        ],
        ["source_1", "source_2", "source_3"],
    )

    compact = compact_research_context(bundle)

    assert compact is not None
    assert [item["source_id"] for item in compact["evidence"]] == [
        "source_1",
        "source_2",
        "source_3",
        "source_1",
    ]


def test_research_context_ranks_evidence_by_query_keyword_overlap() -> None:
    bundle = _bundle_with_evidence(
        [
            ("source_1", "General unrelated background information."),
            ("source_2", "Solar energy can reduce home electricity costs."),
            ("source_3", "Solar panels and solar batteries support home energy."),
            ("source_4", "A different topic about gardening."),
            ("source_5", "Solar"),
        ],
        [f"source_{index}" for index in range(1, 6)],
    )

    compact = compact_research_context(bundle, query_terms=["solar", "home energy"])

    assert compact is not None
    assert [item["source_id"] for item in compact["evidence"]] == [
        "source_2",
        "source_3",
        "source_5",
        "source_1",
    ]


def test_research_context_labels_sources_and_maps_only_included_sources() -> None:
    bundle = _bundle_with_evidence(
        [
            ("source_1", "Solar home energy savings."),
            ("source_2", "Unrelated garden advice."),
            ("source_3", "Solar energy incentives."),
        ],
        ["source_1", "source_2", "source_3"],
    )

    compact = compact_research_context(bundle, ["solar"], labels=True)
    labels = build_research_labels(bundle, ["solar"])

    assert compact is not None
    assert [source["label"] for source in compact["sources"]] == ["S1", "S2", "S3"]
    assert [item["source_id"] for item in compact["evidence"]] == [
        "S1",
        "S3",
        "S2",
    ]
    assert labels == {"S1": "source_1", "S2": "source_2", "S3": "source_3"}


def test_profile_limits_bound_research_and_round_robin_for_sports() -> None:
    bundle = _bundle_with_evidence(
        [
            ("source_1", f"Sports result statistic {index}. " * 20)
            for index in range(5)
        ]
        + [
            ("source_2", f"Historical season record {index}. " * 20)
            for index in range(5)
        ],
        ["source_1", "source_2"],
    )

    compact = compact_research_context(
        bundle,
        query_terms=["sports statistics"],
        profile=get_profile("sports"),
    )

    assert compact is not None
    assert len(compact["evidence"]) == 6
    assert [item["source_id"] for item in compact["evidence"][:2]] == [
        "source_1",
        "source_2",
    ]
    assert all(len(item["text"]) <= 600 for item in compact["evidence"])
