from seo_blog_assistant.schemas import ResearchBundle
from seo_blog_assistant.workflow.research_context import (
    MAX_PROMPT_EVIDENCE_CHARACTERS,
    MAX_PROMPT_EVIDENCE_ITEMS,
    compact_research_context,
)


def test_compact_research_context_bounds_evidence_and_keeps_source_links() -> None:
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
    assert {item["source_id"] for item in compact["evidence"]} == {
        source["source_id"] for source in compact["sources"]
    }


def test_compact_research_context_is_none_without_research() -> None:
    assert compact_research_context(None) is None
