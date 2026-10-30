"""Pluggable web research providers and Tavily's hosted search adapter."""

import hashlib
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from seo_blog_assistant.config import Settings
from seo_blog_assistant.schemas import (
    BlogRequest,
    EvidenceSnippet,
    ResearchBundle,
    ResearchSource,
)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class ResearchProviderError(RuntimeError):
    """Raised when required web research cannot produce usable evidence."""


class ResearchProvider(Protocol):
    """Searches the web and returns traceable source evidence."""

    async def search_and_retrieve(self, request: BlogRequest) -> ResearchBundle: ...


class TavilyResult(BaseModel):
    """Relevant fields from one Tavily Search result."""

    model_config = ConfigDict(extra="ignore")

    url: str
    title: str
    content: str = ""
    raw_content: str | None = None
    published_date: str | None = None


class TavilySearchResponse(BaseModel):
    """Relevant fields from a Tavily Search response."""

    model_config = ConfigDict(extra="ignore")

    results: list[TavilyResult]


class TavilyResearchProvider:
    """Search and retrieve Markdown source content using Tavily's hosted API."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        timeout_seconds: int = 30,
        max_results: int = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.get_secret_value():
            raise ValueError("Tavily API key must not be empty")
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between 1 and 120")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._client = client

    async def search_and_retrieve(self, request: BlogRequest) -> ResearchBundle:
        query = self._build_query(request)
        try:
            response = await self._post_search(query)
            response.raise_for_status()
            payload = TavilySearchResponse.model_validate(response.json())
        except httpx.TimeoutException:
            raise ResearchProviderError("Tavily search timed out.") from None
        except httpx.HTTPStatusError as exc:
            raise ResearchProviderError(
                f"Tavily search failed with HTTP {exc.response.status_code}."
            ) from None
        except httpx.HTTPError:
            raise ResearchProviderError("Tavily search request failed.") from None
        except (ValueError, ValidationError):
            raise ResearchProviderError(
                "Tavily returned an invalid search response."
            ) from None

        sources: list[ResearchSource] = []
        evidence: list[EvidenceSnippet] = []
        seen_source_ids: set[str] = set()
        for result in payload.results:
            content = (result.raw_content or "").strip() or result.content.strip()
            if not content:
                continue
            source_id = self._source_id(result.url)
            if source_id not in seen_source_ids:
                try:
                    sources.append(
                        ResearchSource(
                            source_id=source_id,
                            url=result.url,
                            title=result.title,
                            publisher=self._publisher(result.url),
                            published_at=result.published_date,
                        )
                    )
                except ValidationError:
                    raise ResearchProviderError(
                        "Tavily returned invalid source metadata."
                    ) from None
                seen_source_ids.add(source_id)
            evidence.append(
                EvidenceSnippet(
                    evidence_id=f"evidence_{len(evidence) + 1}",
                    source_id=source_id,
                    text=content[:5000],
                )
            )
        if not sources:
            raise ResearchProviderError(
                "Tavily search completed but returned no usable source evidence."
            )

        return ResearchBundle(
            search_performed=True,
            provider="tavily",
            searched_at=datetime.now(timezone.utc),
            search_queries=[query],
            sources=sources,
            evidence=evidence,
        )

    async def _post_search(self, query: str) -> httpx.Response:
        headers = {
            "Authorization": "Bearer " + self._api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        body = {
            "query": query,
            "search_depth": "basic",
            "max_results": self._max_results,
            "include_raw_content": "markdown",
            "include_answer": False,
            "topic": "general",
        }
        if self._client is not None:
            return await self._client.post(
                TAVILY_SEARCH_URL,
                headers=headers,
                json=body,
                timeout=self._timeout_seconds,
            )
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            return await client.post(TAVILY_SEARCH_URL, headers=headers, json=body)

    @staticmethod
    def _build_query(request: BlogRequest) -> str:
        query = request.topic
        if request.target_country:
            query = f"{query} {request.target_country}"
        return query

    @staticmethod
    def _source_id(url: str) -> str:
        parsed = urlsplit(url)
        canonical_url = urlunsplit(
            (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.query, "")
        )
        digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]
        return f"source_{digest}"

    @staticmethod
    def _publisher(url: str) -> str | None:
        hostname = urlsplit(url).hostname
        if hostname is None:
            return None
        return hostname.removeprefix("www.")


def create_research_provider(settings: Settings) -> ResearchProvider:
    """Create the configured search provider or fail with actionable guidance."""
    if settings.search_provider != "tavily":
        raise ResearchProviderError(
            "Configure SEO_BLOG_SEARCH_PROVIDER=tavily to enable web research."
        )
    if settings.search_api_key is None:
        raise ResearchProviderError(
            "Set SEO_BLOG_SEARCH_API_KEY to enable Tavily web research."
        )
    return TavilyResearchProvider(
        settings.search_api_key,
        timeout_seconds=settings.search_timeout_seconds,
        max_results=settings.search_max_results,
    )
