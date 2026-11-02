"""Shared, validated data contracts for blog generation."""

import base64
import binascii
import re
from datetime import datetime
from io import BytesIO
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from seo_blog_assistant.workflow.topic_profiles import TopicCategory

ResearchMode = Literal["model_knowledge", "web_research"]
ImageGenerationStatus = Literal["skipped", "completed", "failed"]


class AppSchema(BaseModel):
    """Base schema with strict field names and normalized text values."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class BlogRequest(AppSchema):
    """User-provided topic and optional article preferences."""

    topic: str = Field(min_length=3, max_length=300)
    audience: str | None = Field(default=None, min_length=2, max_length=200)
    tone: str = Field(default="professional", min_length=1, max_length=80)
    category: TopicCategory | None = None
    target_country: str | None = Field(default=None, min_length=2, max_length=80)
    target_words: int = Field(default=1200, ge=300, le=5000)
    seo_goals: list[str] = Field(default_factory=list, max_length=12)
    require_freshness: bool = False
    require_citations: bool = False
    generate_images: bool = False


class ResearchDecision(AppSchema):
    """Estimated choice between model knowledge and web research."""

    selected_mode: ResearchMode
    rationale: str = Field(min_length=1, max_length=1000)
    estimated_knowledge_sufficient: bool
    confidence: float = Field(ge=0, le=1)
    topic_category: TopicCategory = "general"
    content_angle: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def check_model_knowledge_estimate(self) -> "ResearchDecision":
        if (
            self.selected_mode == "model_knowledge"
            and not self.estimated_knowledge_sufficient
        ):
            raise ValueError(
                "model_knowledge mode requires an estimate of sufficient knowledge"
            )
        return self


class PlannedSection(AppSchema):
    """Brief and outline for one section; this is not article prose."""

    id: str = Field(min_length=1, max_length=80)
    position: int = Field(ge=0)
    heading: str = Field(min_length=1, max_length=160)
    purpose: str = Field(min_length=1, max_length=1000)
    key_points: list[str] = Field(min_length=1, max_length=12)
    h3_headings: list[str] = Field(min_length=1, max_length=6)
    word_budget: int = Field(ge=50, le=2000)


class SEOPlan(AppSchema):
    """Validated SEO metadata and ordered article outline."""

    title: str = Field(min_length=1, max_length=200)
    meta_description: str = Field(min_length=1, max_length=320)
    search_intent: str = Field(min_length=1, max_length=500)
    primary_keyword: str = Field(min_length=1, max_length=120)
    related_keyword_ideas: list[str] = Field(max_length=30)
    target_words: int = Field(ge=300, le=5000)
    introduction_brief: str = Field(min_length=1, max_length=2000)
    introduction_word_budget: int = Field(ge=50, le=500)
    sections: list[PlannedSection] = Field(min_length=3, max_length=10)
    conclusion_brief: str = Field(min_length=1, max_length=2000)
    conclusion_word_budget: int = Field(ge=50, le=500)

    @model_validator(mode="before")
    @classmethod
    def normalize_section_positions(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        sections = value.get("sections")
        if not isinstance(sections, list):
            return value

        normalized_value = value.copy()
        normalized_sections = []
        for position, section in enumerate(sections):
            if isinstance(section, dict):
                normalized_section = section.copy()
                normalized_section["position"] = position
                normalized_sections.append(normalized_section)
            else:
                normalized_sections.append(section)
        normalized_value["sections"] = normalized_sections
        return normalized_value

    @model_validator(mode="before")
    @classmethod
    def rescale_word_budgets(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        target_words = value.get("target_words")
        sections = value.get("sections")
        budget_values = [
            value.get("introduction_word_budget"),
            value.get("conclusion_word_budget"),
        ]
        if (
            not isinstance(sections, list)
            or not sections
            or type(target_words) is not int
        ):
            return value
        if any(type(budget) is not int for budget in budget_values):
            return value
        if any(not isinstance(section, dict) for section in sections):
            return value
        section_budgets = [section.get("word_budget") for section in sections]
        if any(type(budget) is not int for budget in section_budgets):
            return value

        budgets = [budget_values[0], *section_budgets, budget_values[1]]
        current_total = sum(budgets)
        tolerance = max(50, round(target_words * 0.1))
        if current_total <= 0 or abs(current_total - target_words) <= tolerance:
            return value

        scaled = [
            round(budget * target_words / current_total)
            for budget in budgets
        ]
        bounds = [(50, 500), *[(50, 2000) for _ in sections], (50, 500)]
        clamped = [
            min(max(word_count, minimum), maximum)
            for word_count, (minimum, maximum) in zip(scaled, bounds)
        ]
        remainder = target_words - sum(clamped)
        largest_section_index = max(
            range(len(sections)),
            key=lambda index: budgets[index + 1],
        ) + 1
        minimum, maximum = bounds[largest_section_index]
        adjusted_largest = clamped[largest_section_index] + remainder
        if not minimum <= adjusted_largest <= maximum:
            return value
        clamped[largest_section_index] = adjusted_largest
        if sum(clamped) != target_words:
            return value

        normalized_value = value.copy()
        normalized_value["introduction_word_budget"] = clamped[0]
        normalized_value["conclusion_word_budget"] = clamped[-1]
        normalized_sections = []
        for section, word_budget in zip(sections, clamped[1:-1]):
            normalized_section = section.copy()
            normalized_section["word_budget"] = word_budget
            normalized_sections.append(normalized_section)
        normalized_value["sections"] = normalized_sections
        return normalized_value

    @model_validator(mode="after")
    def check_section_identity_and_order(self) -> "SEOPlan":
        ids = [section.id for section in self.sections]
        positions = [section.position for section in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError("section IDs must be unique")
        if positions != list(range(len(self.sections))):
            raise ValueError("section positions must match their outline order")
        planned_words = (
            self.introduction_word_budget
            + self.conclusion_word_budget
            + sum(section.word_budget for section in self.sections)
        )
        tolerance = max(50, round(self.target_words * 0.1))
        if abs(planned_words - self.target_words) > tolerance:
            raise ValueError(
                "word budgets must sum to within 10% (or 50 words) of target_words"
            )
        return self


class DraftSubsection(AppSchema):
    """Finished prose for one planned H3 heading."""

    heading: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list, max_length=30)


class SectionDraft(AppSchema):
    """Finished prose returned by one section-writing worker."""

    task_id: str = Field(min_length=1, max_length=80)
    position: int = Field(ge=0)
    heading: str = Field(min_length=1, max_length=160)
    opening: str = Field(min_length=1)
    subsections: list[DraftSubsection] = Field(min_length=1, max_length=6)
    closing_insight: str = Field(min_length=1)
    key_points_covered: list[str] = Field(min_length=1, max_length=12)
    image_prompt: str | None = Field(default=None, max_length=400)


class GeneratedImage(AppSchema):
    """Validated PNG image generated for an article."""

    model_name: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=2000)
    alt_text: str = Field(min_length=1, max_length=500)
    media_type: Literal["image/png"] = "image/png"
    image_base64: str = Field(min_length=1, max_length=20_000_000)

    @model_validator(mode="after")
    def check_png_data(self) -> "GeneratedImage":
        try:
            image_data = base64.b64decode(self.image_base64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("image_base64 must contain valid base64 data") from None
        if len(image_data) > 15_000_000:
            raise ValueError("generated image exceeds the 15 MB limit")

        try:
            with Image.open(BytesIO(image_data)) as image:
                if image.format != "PNG":
                    raise ValueError("generated image must be a PNG")
                if image.width > 8192 or image.height > 8192:
                    raise ValueError("generated image dimensions exceed the limit")
                if image.width * image.height > 40_000_000:
                    raise ValueError("generated image dimensions exceed the limit")
                image.verify()
        except (
            Image.DecompressionBombError,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
        ):
            raise ValueError("image_base64 must contain a valid PNG image") from None
        return self


class ImageGenerationResult(AppSchema):
    """Image branch status without hiding optional provider failures."""

    status: ImageGenerationStatus
    image: GeneratedImage | None = None
    message: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def check_result_consistency(self) -> "ImageGenerationResult":
        if self.status == "completed" and self.image is None:
            raise ValueError("completed image generation requires an image")
        if self.status != "completed" and self.image is not None:
            raise ValueError("only completed image generation can include an image")
        if self.status == "failed" and self.message is None:
            raise ValueError("failed image generation requires a message")
        return self


class InlineImage(AppSchema):
    """Image generated for a validated inline Markdown placeholder."""

    image_id: str = Field(pattern=r"^img_[1-4]$")
    after_heading: str = Field(min_length=1, max_length=160)
    alt_text: str = Field(min_length=1, max_length=500)
    status: Literal["completed", "failed"]
    image: GeneratedImage | None = None
    message: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def check_result_consistency(self) -> "InlineImage":
        if self.status == "completed" and self.image is None:
            raise ValueError("completed inline image generation requires an image")
        if self.status == "failed" and self.image is not None:
            raise ValueError("failed inline image generation cannot include an image")
        if self.status == "failed" and self.message is None:
            raise ValueError("failed inline image generation requires a message")
        return self


class ImagePlaceholder(AppSchema):
    """Copyable prompt for an article image generated outside the application."""

    image_id: str = Field(pattern=r"^img_[1-4]$")
    after_heading: str = Field(min_length=1, max_length=160)
    prompt: str = Field(min_length=1, max_length=400)
    alt_text: str = Field(min_length=1, max_length=500)


class ArticleBookends(AppSchema):
    """Finished introduction and conclusion prose, distinct from their briefs."""

    introduction: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)


class ArticleResult(AppSchema):
    """Validated article content and metadata returned by the workflow."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    title: str = Field(min_length=1, max_length=200)
    meta_description: str = Field(min_length=1, max_length=320)
    search_intent: str = Field(min_length=1, max_length=500)
    primary_keyword: str = Field(min_length=1, max_length=120)
    related_keyword_ideas: list[str] = Field(max_length=30)
    research_mode: ResearchMode | None = None
    topic_category: TopicCategory = "general"
    content_angle: str | None = Field(default=None, max_length=200)
    markdown: str = Field(min_length=1)
    sources: list["ResearchSource"] = Field(max_length=30)
    image_result: ImageGenerationResult
    inline_images: list[InlineImage] = Field(default_factory=list, max_length=4)
    image_placeholders: list[ImagePlaceholder] = Field(
        default_factory=list,
        max_length=4,
    )
    warnings: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def check_markdown_sources(self) -> "ArticleResult":
        if not self.markdown.startswith(f"# {self.title}\n"):
            raise ValueError("article Markdown must begin with the article title")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("article source IDs must be unique")
        citations = set(re.findall(r"\[(source_[A-Za-z0-9_-]+)\]", self.markdown))
        if citations - set(source_ids):
            raise ValueError("article Markdown citations must reference listed sources")
        inline_image_ids = [image.image_id for image in self.inline_images]
        if len(inline_image_ids) != len(set(inline_image_ids)):
            raise ValueError("inline image IDs must be unique")
        markdown_image_ids = set(
            re.findall(r"\]\(image:(img_[1-4])\)", self.markdown)
        )
        completed_image_ids = {
            image.image_id for image in self.inline_images if image.status == "completed"
        }
        if markdown_image_ids != completed_image_ids:
            raise ValueError(
                "article Markdown image placeholders must match completed inline images"
            )
        placeholder_ids = [item.image_id for item in self.image_placeholders]
        if len(placeholder_ids) != len(set(placeholder_ids)):
            raise ValueError("article image placeholder IDs must be unique")
        markdown_placeholder_ids = re.findall(
            r"(?m)^> 🖼️ \*\*IMAGE PLACEHOLDER \((img_[1-4])\):",
            self.markdown,
        )
        if markdown_placeholder_ids != placeholder_ids:
            raise ValueError(
                "article image placeholder blocks must match their metadata in order"
            )
        if placeholder_ids != [
            f"img_{index}" for index in range(1, len(placeholder_ids) + 1)
        ]:
            raise ValueError("article image placeholder IDs must follow document order")
        return self


class ResearchSource(AppSchema):
    """Source metadata returned by a successful web-research operation."""

    source_id: str = Field(min_length=1, max_length=80)
    url: HttpUrl
    title: str = Field(min_length=1, max_length=500)
    publisher: str | None = Field(default=None, max_length=200)
    published_at: str | None = Field(default=None, max_length=80)


class EvidenceSnippet(AppSchema):
    """Extracted evidence traceable to a retrieved source."""

    evidence_id: str = Field(min_length=1, max_length=80)
    source_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=5000)


class ResearchBundle(AppSchema):
    """Evidence from a search operation that actually completed."""

    search_performed: Literal[True]
    provider: str = Field(min_length=1, max_length=80)
    searched_at: datetime
    search_queries: list[str] = Field(min_length=1, max_length=10)
    warnings: list[str] = Field(default_factory=list, max_length=10)
    sources: list[ResearchSource] = Field(min_length=1, max_length=30)
    evidence: list[EvidenceSnippet] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def check_research_references(self) -> "ResearchBundle":
        source_ids = [source.source_id for source in self.sources]
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("research source IDs must be unique")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        unknown_source_ids = {item.source_id for item in self.evidence} - set(source_ids)
        if unknown_source_ids:
            raise ValueError("evidence must reference a source in the research bundle")
        if self.searched_at.tzinfo is None:
            raise ValueError("searched_at must include a timezone")
        return self
