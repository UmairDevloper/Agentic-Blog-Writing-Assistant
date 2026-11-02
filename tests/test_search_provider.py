import asyncio
import json
from datetime import timezone

import httpx
import pytest
from pydantic import SecretStr

from seo_blog_assistant.config import Settings
from seo_blog_assistant.providers.search import (
    ResearchProviderError,
    TavilyResearchProvider,
    create_research_provider,
)
from seo_blog_assistant.schemas import BlogRequest
from seo_blog_assistant.workflow.topic_profiles import get_profile


def test_tavily_search_returns_sources_and_retrieved_evidence() -> None:
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://www.example.com/guide?ref=search#section",
                        "title": "A useful guide",
                        "content": "Search result excerpt.",
                        "raw_content": "# Retrieved article\n\nFull relevant text.",
                        "published_date": "2026-10-01",
                        "score": 0.9,
                    }
                ]
            },
        )

    async def search() -> tuple[object, httpx.Request]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-test-secret"),
                timeout_seconds=17,
                max_results=3,
                client=client,
            )
            result = await provider.search_and_retrieve(
                BlogRequest(
                    topic="Home energy savings",
                    target_country="Canada",
                )
            )
            return result, requests[0]

    bundle, request = asyncio.run(search())
    body = json.loads(request.content)

    assert request.method == "POST"
    assert request.url == "https://api.tavily.com/search"
    assert request.headers["Authorization"] == "Bearer tvly-test-secret"
    assert "tvly-test-secret" not in request.content.decode()
    assert body["query"] == "Home energy savings Canada"
    assert body["max_results"] == 3
    assert body["include_raw_content"] == "markdown"
    assert bundle.search_performed is True
    assert bundle.provider == "tavily"
    assert bundle.search_queries == ["Home energy savings Canada"]
    assert bundle.sources[0].source_id.startswith("source_")
    assert str(bundle.sources[0].url) == (
        "https://www.example.com/guide?ref=search#section"
    )
    assert bundle.sources[0].publisher == "example.com"
    assert bundle.sources[0].published_at == "2026-10-01"
    assert bundle.evidence[0].source_id == bundle.sources[0].source_id
    assert "Full relevant text." in bundle.evidence[0].text
    assert bundle.searched_at.tzinfo == timezone.utc


def test_tavily_http_error_is_explicit_and_does_not_expose_token() -> None:
    def handle_request(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid tvly-private-secret")

    async def search() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-private-secret"),
                client=client,
            )
            await provider.search_and_retrieve(BlogRequest(topic="A useful topic"))

    with pytest.raises(ResearchProviderError, match="API key") as error:
        asyncio.run(search())

    assert "tvly-private-secret" not in str(error.value)


def test_tavily_empty_evidence_is_not_reported_as_success() -> None:
    def handle_request(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/empty",
                        "title": "Empty result",
                        "content": "",
                    }
                ]
            },
        )

    async def search() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-test-secret"),
                client=client,
            )
            await provider.search_and_retrieve(BlogRequest(topic="A useful topic"))

    with pytest.raises(ResearchProviderError, match="no usable source evidence"):
        asyncio.run(search())


def test_tavily_timeout_is_reported_without_echoing_exception_details() -> None:
    def handle_request(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("request failed with tvly-private-secret")

    async def search() -> None:
        async def no_sleep(_delay: float) -> None:
            return None

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-private-secret"),
                sleep=no_sleep,
                client=client,
            )
            await provider.search_and_retrieve(BlogRequest(topic="A useful topic"))

    with pytest.raises(ResearchProviderError, match="timed out") as error:
        asyncio.run(search())

    assert "tvly-private-secret" not in str(error.value)


def test_tavily_invalid_source_url_is_not_returned_as_research() -> None:
    def handle_request(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "not-a-url",
                        "title": "Invalid source",
                        "content": "Some content",
                    }
                ]
            },
        )

    async def search() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-test-secret"),
                client=client,
            )
            await provider.search_and_retrieve(BlogRequest(topic="A useful topic"))

    with pytest.raises(ResearchProviderError, match="invalid source metadata"):
        asyncio.run(search())


def test_search_provider_factory_requires_tavily_credentials() -> None:
    with pytest.raises(ResearchProviderError, match="SEARCH_PROVIDER=tavily"):
        create_research_provider(Settings(_env_file=None))

    with pytest.raises(ResearchProviderError, match="SEARCH_API_KEY"):
        create_research_provider(
            Settings(_env_file=None, search_provider="tavily")
        )

    provider = create_research_provider(
        Settings(
            _env_file=None,
            search_provider="tavily",
            search_api_key=SecretStr("tvly-test-secret"),
        )
    )
    assert isinstance(provider, TavilyResearchProvider)


def test_tavily_prefers_long_query_relevant_content_over_raw_content() -> None:
    relevant_content = "Relevant extract. " * 20

    def handle_request(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/guide",
                        "title": "A useful guide",
                        "content": relevant_content,
                        "raw_content": "Navigation and unrelated page content.",
                    }
                ]
            },
        )

    async def search():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-test-secret"),
                client=client,
            )
            return await provider.search_and_retrieve(
                BlogRequest(topic="A useful topic")
            )

    bundle = asyncio.run(search())
    assert bundle.evidence[0].text == relevant_content.strip()


def test_tavily_trims_leading_boilerplate_from_raw_content() -> None:
    useful_line = "Substantive retrieved evidence " + ("about the topic " * 8)

    def handle_request(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/guide",
                        "title": "A useful guide",
                        "content": "Short excerpt.",
                        "raw_content": f"Home\nMenu\n\n{useful_line}\nMore detail.",
                    }
                ]
            },
        )

    async def search():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-test-secret"),
                client=client,
            )
            return await provider.search_and_retrieve(
                BlogRequest(topic="A useful topic")
            )

    bundle = asyncio.run(search())
    assert bundle.evidence[0].text.startswith(useful_line)
    assert "Home" not in bundle.evidence[0].text


def test_tavily_retries_server_error_then_returns_evidence() -> None:
    request_count = 0
    sleep_calls: list[float] = []

    def handle_request(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/guide",
                        "title": "A useful guide",
                        "content": "Useful evidence from the source.",
                    }
                ]
            },
        )

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    async def search():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-test-secret"),
                sleep=fake_sleep,
                client=client,
            )
            return await provider.search_and_retrieve(
                BlogRequest(topic="A useful topic")
            )

    bundle = asyncio.run(search())
    assert bundle.search_performed
    assert request_count == 2
    assert len(sleep_calls) == 1


def test_tavily_maps_quota_response_without_exposing_provider_body() -> None:
    request_count = 0

    def handle_request(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(429, text="private quota details")

    async def no_sleep(_delay: float) -> None:
        return None

    async def search() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-test-secret"),
                max_attempts=1,
                sleep=no_sleep,
                client=client,
            )
            await provider.search_and_retrieve(BlogRequest(topic="A useful topic"))

    with pytest.raises(ResearchProviderError, match="quota or plan limit") as error:
        asyncio.run(search())

    assert request_count == 1
    assert "private quota details" not in str(error.value)


def test_sports_search_uses_two_queries_deduplicates_and_news_first() -> None:
    request_bodies: list[dict[str, object]] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_bodies.append(body)
        results = (
            [
                {
                    "url": "https://example.com/match#recent",
                    "title": "Recent match",
                    "content": "Recent match score and current standings evidence.",
                },
                {
                    "url": "https://example.com/record",
                    "title": "Sports record",
                    "content": "Historical record and season evidence.",
                },
            ]
            if len(request_bodies) == 1
            else [
                {
                    "url": "https://example.com/match#history",
                    "title": "Duplicate match",
                    "content": "Duplicate page with another fragment.",
                },
                {
                    "url": "https://example.com/history",
                    "title": "Past seasons",
                    "content": "Past seasons and historical comparison evidence.",
                },
            ]
        )
        return httpx.Response(200, json={"results": results})

    async def search():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-test-secret"),
                max_results=3,
                client=client,
            )
            return await provider.search_and_retrieve(
                BlogRequest(
                    topic="Harbor City football",
                    target_country="Canada",
                    require_freshness=True,
                ),
                get_profile("sports"),
            )

    bundle = asyncio.run(search())

    assert bundle.search_queries == [
        "Harbor City football latest results stats Canada",
        "Harbor City football history records past seasons Canada",
    ]
    assert [body["topic"] for body in request_bodies] == ["news", "general"]
    assert len(bundle.sources) == 3
    assert len({source.source_id for source in bundle.sources}) == 3
    assert len(bundle.evidence) == 3


def test_failed_follow_up_search_keeps_first_query_evidence_and_warns() -> None:
    request_count = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 2:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/match",
                        "title": "Match result",
                        "content": "The first query returned useful sports evidence.",
                    }
                ]
            },
        )

    async def search():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-test-secret"),
                max_attempts=1,
                client=client,
            )
            return await provider.search_and_retrieve(
                BlogRequest(topic="Harbor City football"),
                get_profile("sports"),
            )

    bundle = asyncio.run(search())

    assert request_count == 2
    assert len(bundle.search_queries) == 2
    assert len(bundle.sources) == 1
    assert len(bundle.evidence) == 1
    assert len(bundle.warnings) == 1
    assert "evidence from earlier queries was retained" in bundle.warnings[0]
