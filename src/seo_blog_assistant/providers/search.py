"""Pluggable web research providers and Tavily's hosted search adapter."""

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from seo_blog_assistant.config import Settings
from seo_blog_assistant.providers.retry import RetryDecision, retry_async
from seo_blog_assistant.schemas import (
    BlogRequest,
    EvidenceSnippet,
    ResearchBundle,
    ResearchSource,
)
from seo_blog_assistant.workflow.topic_profiles import TopicProfile, get_profile

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
logger = logging.getLogger(__name__)


class ResearchProviderError(RuntimeError):
    """Raised when required web research cannot produce usable evidence."""


class ResearchProvider(Protocol):
    """Searches the web and returns traceable source evidence."""

    async def search_and_retrieve(
        self,
        request: BlogRequest,
        profile: TopicProfile,
    ) -> ResearchBundle: ...


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
        max_attempts: int = 3,
        queries_per_article: int = 2,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.get_secret_value():
            raise ValueError("Tavily API key must not be empty")
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between 1 and 120")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if not 1 <= queries_per_article <= 3:
            raise ValueError("queries_per_article must be between 1 and 3")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._max_attempts = max_attempts
        self._queries_per_article = queries_per_article
        self._sleep = sleep
        self._client = client

    async def search_and_retrieve(
        self,
        request: BlogRequest,
        profile: TopicProfile | None = None,
    ) -> ResearchBundle:
        selected_profile = profile or get_profile(request.category)
        queries = self._build_queries(request, selected_profile)
        sources: list[ResearchSource] = []
        evidence: list[EvidenceSnippet] = []
        seen_source_ids: set[str] = set()
        warnings: list[str] = []

        for query_index, query in enumerate(queries):
            topic = (
                "news"
                if query_index == 0
                and request.require_freshness
                and selected_profile.category in {"sports", "politics"}
                else "general"
            )
            try:
                payload = await self._search_query(query, topic=topic)
            except ResearchProviderError:
                if query_index == 0:
                    raise
                warnings.append(
                    f"Search query {query_index + 1} failed; evidence from earlier "
                    "queries was retained."
                )
                logger.warning(
                    "Search query failed after an earlier query succeeded "
                    "query_index=%s",
                    query_index + 1,
                )
                continue

            for result in payload.results:
                source_id = self._source_id(result.url)
                if source_id in seen_source_ids or len(sources) >= self._max_results:
                    continue
                content = _result_evidence_text(result)
                if not content:
                    continue
                try:
                    source = ResearchSource(
                        source_id=source_id,
                        url=result.url,
                        title=result.title,
                        publisher=self._publisher(result.url),
                        published_at=result.published_date,
                    )
                except ValidationError:
                    raise ResearchProviderError(
                        "Tavily returned invalid source metadata."
                    ) from None
                sources.append(source)
                evidence.append(
                    EvidenceSnippet(
                        evidence_id=f"evidence_{len(evidence) + 1}",
                        source_id=source_id,
                        text=content[:5000],
                    )
                )
                seen_source_ids.add(source_id)

        if not sources:
            raise ResearchProviderError(
                "Tavily search completed but returned no usable source evidence."
            )

        return ResearchBundle(
            search_performed=True,
            provider="tavily",
            searched_at=datetime.now(timezone.utc),
            search_queries=queries,
            sources=sources,
            evidence=evidence,
            warnings=warnings,
        )

    async def _search_query(
        self,
        query: str,
        *,
        topic: str,
    ) -> TavilySearchResponse:
        attempts = 0

        async def make_request() -> httpx.Response:
            nonlocal attempts
            attempts += 1
            logger.info(
                "Calling search provider provider=tavily endpoint=search attempt=%s",
                attempts,
            )
            try:
                response = await self._post_search(query, topic=topic)
                response.raise_for_status()
            except Exception as exc:
                status_code = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                logger.warning(
                    "Search provider attempt failed provider=tavily endpoint=search "
                    "attempt=%s status=%s error_type=%s",
                    attempts,
                    status_code,
                    type(exc).__name__,
                )
                raise
            logger.info(
                "Search provider attempt succeeded provider=tavily endpoint=search "
                "attempt=%s status=%s",
                attempts,
                response.status_code,
            )
            return response

        def classify(exc: Exception) -> RetryDecision | None:
            if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                return RetryDecision()
            if not isinstance(exc, httpx.HTTPStatusError):
                return None
            status_code = exc.response.status_code
            if status_code in {500, 502, 503, 504}:
                return RetryDecision()
            if status_code == 429:
                retry_after = _retry_after_seconds(exc.response)
                if retry_after is not None and retry_after > 60:
                    return None
                return RetryDecision(
                    delay_seconds=float(retry_after) if retry_after is not None else None
                )
            return None

        try:
            response = await retry_async(
                make_request,
                max_attempts=self._max_attempts,
                classify=classify,
                sleep=self._sleep,
                logger=logger,
                label="tavily:search",
            )
        except Exception as exc:
            raise _map_search_error(exc, attempts=attempts) from None

        try:
            return TavilySearchResponse.model_validate(response.json())
        except (ValueError, ValidationError):
            raise ResearchProviderError(
                "Tavily returned an invalid search response."
            ) from None

    async def _post_search(self, query: str, *, topic: str) -> httpx.Response:
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
            "topic": topic,
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

    def _build_queries(
        self,
        request: BlogRequest,
        profile: TopicProfile,
    ) -> list[str]:
        hints = profile.search_hints[: self._queries_per_article]
        query_parts = [f"{request.topic} {hint}" for hint in hints] or [request.topic]
        if request.target_country:
            query_parts = [
                f"{query} {request.target_country}" for query in query_parts
            ]
        return query_parts

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
        max_attempts=settings.search_max_attempts,
        queries_per_article=settings.search_queries_per_article,
    )


def _retry_after_seconds(response: httpx.Response) -> int | None:
    try:
        value = response.headers.get("retry-after")
        return max(0, int(float(value))) if value is not None else None
    except (TypeError, ValueError):
        return None


def _map_search_error(exc: Exception, *, attempts: int) -> ResearchProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ResearchProviderError(
            f"Tavily search timed out after {attempts} attempts."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code in {401, 403}:
            return ResearchProviderError(
                "Tavily rejected the API key; check the search credentials."
            )
        if status_code in {429, 432, 433}:
            return ResearchProviderError(
                "Tavily quota or plan limit reached; check the account limits "
                "before retrying."
            )
        if status_code >= 500:
            return ResearchProviderError(
                f"Tavily upstream error with HTTP {status_code} after "
                f"{attempts} attempts."
            )
        return ResearchProviderError(
            f"Tavily search failed with HTTP {status_code}."
        )
    if isinstance(exc, httpx.TransportError):
        return ResearchProviderError(
            f"Could not reach Tavily after {attempts} attempts."
        )
    return ResearchProviderError("Tavily search request failed.")


def _result_evidence_text(result: TavilyResult) -> str:
    content = result.content.strip()
    if len(content) >= 200:
        return content[:5000]

    raw_content = result.raw_content or ""
    lines = raw_content.splitlines()
    first_substantive = next(
        (index for index, line in enumerate(lines) if len(line.strip()) >= 80),
        None,
    )
    if first_substantive is not None:
        raw_content = "\n".join(lines[first_substantive:])
    else:
        raw_content = "\n".join(lines)
    raw_content = raw_content.strip()
    if raw_content:
        return raw_content[:5000]
    return content[:5000]
