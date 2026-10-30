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

    with pytest.raises(ResearchProviderError, match="HTTP 401") as error:
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
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request)
        ) as client:
            provider = TavilyResearchProvider(
                SecretStr("tvly-private-secret"),
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
