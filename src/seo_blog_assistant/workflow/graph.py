"""LangGraph construction for research, planning, and article writing."""

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from seo_blog_assistant.config import Settings
from seo_blog_assistant.providers.image import ImageProvider, ImageProviderError
from seo_blog_assistant.providers.search import (
    ResearchProvider,
    ResearchProviderError,
)
from seo_blog_assistant.providers.text_model import StructuredTextModel
from seo_blog_assistant.schemas import (
    ArticleBookends,
    ArticleResult,
    BlogRequest,
    ImageGenerationResult,
    ResearchBundle,
    ResearchMode,
    SEOPlan,
    SectionDraft,
)
from seo_blog_assistant.workflow.fact_guard import find_unsupported_figures
from seo_blog_assistant.workflow.assembly import (
    assemble_article,
    assemble_batched_article,
    write_bookends,
)
from seo_blog_assistant.workflow.batched_writing import write_article_passes
from seo_blog_assistant.workflow.inline_images import (
    build_image_placeholders,
    build_section_image_placeholders,
    generate_inline_images,
)
from seo_blog_assistant.workflow.planning import create_seo_plan
from seo_blog_assistant.workflow.research_context import compact_research_context
from seo_blog_assistant.workflow.research import assess_research, research_route
from seo_blog_assistant.workflow.state import BlogWorkflowState
from seo_blog_assistant.workflow.topic_profiles import TopicProfile, get_profile
from seo_blog_assistant.workflow.writing import write_section

logger = logging.getLogger(__name__)


def build_research_planning_graph(
    model: StructuredTextModel,
    search_provider: ResearchProvider | None = None,
) -> CompiledStateGraph[
    BlogWorkflowState,
    None,
    BlogWorkflowState,
    BlogWorkflowState,
]:
    """Build the research-and-planning segment with explicit routing."""
    return _build_graph(model, search_provider, write_sections=False)


def build_blog_workflow_graph(
    model: StructuredTextModel,
    search_provider: ResearchProvider | None = None,
    image_provider: ImageProvider | None = None,
    *,
    writing_mode: Literal["batched", "per_section"] = "batched",
    max_words_per_call: int = 1500,
    max_images: int = 3,
    settings: Settings | None = None,
) -> CompiledStateGraph[
    BlogWorkflowState,
    None,
    BlogWorkflowState,
    BlogWorkflowState,
]:
    """Build the research, planning, and selected article-writing workflow."""
    return _build_graph(
        model,
        search_provider,
        image_provider,
        write_sections=True,
        writing_mode=writing_mode,
        max_words_per_call=max_words_per_call,
        max_images=max_images,
        settings=settings,
    )


def _build_graph(
    model: StructuredTextModel,
    search_provider: ResearchProvider | None,
    image_provider: ImageProvider | None = None,
    *,
    write_sections: bool,
    writing_mode: Literal["batched", "per_section"] = "batched",
    max_words_per_call: int = 1500,
    max_images: int = 3,
    settings: Settings | None = None,
) -> CompiledStateGraph[
    BlogWorkflowState,
    None,
    BlogWorkflowState,
    BlogWorkflowState,
]:
    if writing_mode not in {"batched", "per_section"}:
        raise ValueError("writing_mode must be 'batched' or 'per_section'")
    runtime_settings = settings or Settings(
        _env_file=None,
        writing_mode=writing_mode,
        max_words_per_call=max_words_per_call,
        max_images=max_images,
    )

    async def assess_node(state: BlogWorkflowState) -> dict[str, object]:
        decision = await assess_research(
            state["request"],
            model,
            search_provider_configured=search_provider is not None,
        )
        profile = get_profile(decision.topic_category)
        warnings: list[str] = []
        if (
            profile.research_policy == "recommended"
            and search_provider is None
            and not state["request"].require_freshness
            and not state["request"].require_citations
        ):
            warnings.append(
                "Research was recommended for this topic but no search provider "
                "is configured; figures are unverified."
            )
        return {
            "research_decision": decision,
            "topic_category": decision.topic_category,
            "content_angle": decision.content_angle,
            "warnings": warnings,
        }

    def route_node(state: BlogWorkflowState) -> ResearchMode:
        route = research_route(state["research_decision"])
        if route == "web_research" and search_provider is None:
            profile = get_profile(state["research_decision"].topic_category)
            if profile.research_policy == "required":
                fact_reason = {
                    "sports": "stats and records",
                    "politics": "results and public claims",
                    "business_finance": "figures and market claims",
                }.get(profile.category, "important facts")
                raise ResearchProviderError(
                    f"{profile.label} articles need web research for {fact_reason}. "
                    "Set SEO_BLOG_SEARCH_PROVIDER and "
                    "SEO_BLOG_SEARCH_API_KEY."
                )
            raise ResearchProviderError(
                "Web research was selected, but no search provider is configured."
            )
        return route

    async def start_research_node(state: BlogWorkflowState) -> dict[str, str]:
        if state["research_decision"].selected_mode != "web_research":
            raise ResearchProviderError(
                "The research status cannot start outside web_research mode."
            )
        return {"workflow_stage": "research_started"}

    async def research_node(state: BlogWorkflowState) -> dict[str, object]:
        if search_provider is None:
            raise ResearchProviderError(
                "Web research was selected, but no search provider is configured."
            )
        profile = get_profile(state["research_decision"].topic_category)
        bundle = await search_provider.search_and_retrieve(
            state["request"],
            profile,
        )
        return {
            "research_bundle": bundle,
            "warnings": bundle.warnings,
            "workflow_stage": "research_completed",
        }

    async def create_plan_node(state: BlogWorkflowState) -> dict[str, object]:
        plan = await create_seo_plan(
            request=state["request"],
            decision=state["research_decision"],
            model=model,
            research=state.get("research_bundle"),
        )
        return {"seo_plan": plan, "workflow_stage": "plan_ready"}

    def dispatch_sections(state: BlogWorkflowState) -> list[Send]:
        sends = [
            Send(
                "write_section",
                {
                    "request": state["request"],
                    "research_decision": state["research_decision"],
                    "research_bundle": state.get("research_bundle"),
                    "seo_plan": state["seo_plan"],
                    "active_section": section,
                },
            )
            for section in state["seo_plan"].sections
        ]
        if runtime_settings.image_mode == "generate":
            sends.append(
                Send(
                    "images",
                    {
                        "request": state["request"],
                        "seo_plan": state["seo_plan"],
                    },
                )
            )
        return sends

    async def write_section_node(state: BlogWorkflowState) -> dict[str, object]:
        draft, warnings = await write_section(
            request=state["request"],
            plan=state["seo_plan"],
            section=state["active_section"],
            model=model,
            research=state.get("research_bundle"),
            image_mode=runtime_settings.image_mode,
            profile=get_profile(state["research_decision"].topic_category),
        )
        return {"section_drafts": [draft], "warnings": warnings}

    def collect_sections_node(
        state: BlogWorkflowState,
    ) -> dict[str, list[SectionDraft]]:
        plan = state["seo_plan"]
        drafts = state.get("section_drafts", [])
        if len(drafts) != len(plan.sections):
            raise ValueError("section writing did not return one draft per planned section")
        drafts_by_id = {draft.task_id: draft for draft in drafts}
        expected_ids = {section.id for section in plan.sections}
        if len(drafts_by_id) != len(drafts) or set(drafts_by_id) != expected_ids:
            raise ValueError("section drafts do not match the planned section identities")
        return {"ordered_section_drafts": sorted(drafts, key=lambda draft: draft.position)}

    async def images_node(state: BlogWorkflowState) -> dict[str, object]:
        if not state["request"].generate_images:
            result = ImageGenerationResult(
                status="skipped",
                message="Image generation was not requested.",
            )
        elif max_images == 0:
            result = ImageGenerationResult(
                status="skipped",
                message="Image generation is disabled by the configured image limit.",
            )
        elif image_provider is None:
            result = ImageGenerationResult(
                status="failed",
                message=(
                    "Image generation was requested, but no image provider is configured."
                ),
            )
        else:
            plan = state["seo_plan"]
            prompt = (
                "Create an illustrative or abstract scene and visual metaphor for "
                "a blog article; do not make a diagram, chart, flowchart, screenshot, "
                "or UI. Topic: "
                f"{state['request'].topic!r}. Title: {plan.title!r}. "
                "Clean editorial illustration, flat colors, no text. No letters, "
                "numbers, logos, or watermarks."
            )
            try:
                image = await image_provider.generate_image(
                    prompt,
                    f"Illustration for {plan.title}",
                )
            except ImageProviderError as exc:
                result = ImageGenerationResult(
                    status="failed",
                    message=str(exc)[:500],
                )
            except Exception as exc:
                safe_exception = RuntimeError(type(exc).__name__)
                logger.exception(
                    "Unexpected optional image generation failure",
                    exc_info=(RuntimeError, safe_exception, exc.__traceback__),
                )
                result = ImageGenerationResult(
                    status="failed",
                    message=(
                        "Image generation failed unexpectedly; article text is "
                        "available."
                    ),
                )
            else:
                result = ImageGenerationResult(status="completed", image=image)
        return {
            "image_result": result,
            "workflow_stage": f"images_{result.status}",
        }

    def prepare_section_placeholders_node(
        state: BlogWorkflowState,
    ) -> dict[str, object]:
        generated = build_section_image_placeholders(
            plan=state["seo_plan"],
            section_drafts=state["ordered_section_drafts"],
            generate_images=state["request"].generate_images,
            max_images=max_images,
        )
        return {
            "image_placeholders": generated.image_placeholders,
            "image_result": generated.image_result,
            "warnings": generated.warnings,
        }

    async def assemble_article_node(
        state: BlogWorkflowState,
    ) -> dict[str, object]:
        bookends: ArticleBookends = await write_bookends(
            request=state["request"],
            plan=state["seo_plan"],
            model=model,
            research=state.get("research_bundle"),
            profile=get_profile(state["research_decision"].topic_category),
            content_angle=state["research_decision"].content_angle,
        )
        profile = get_profile(state["research_decision"].topic_category)
        article: ArticleResult = assemble_article(
            plan=state["seo_plan"],
            bookends=bookends,
            section_drafts=state["ordered_section_drafts"],
            research=state.get("research_bundle"),
            image_result=state.get("image_result"),
            image_placeholders=state.get("image_placeholders", []),
            research_mode=state["research_decision"].selected_mode,
            topic_category=profile.category,
            content_angle=state["research_decision"].content_angle,
            warnings=state.get("warnings", []),
        )
        article = _apply_fact_guard(
            article,
            request=state["request"],
            plan=state["seo_plan"],
            profile=profile,
            research=state.get("research_bundle"),
            writing_mode=writing_mode,
        )
        return {
            "bookends": bookends,
            "article": article,
            "image_result": article.image_result,
            "image_placeholders": article.image_placeholders,
            "inline_images": article.inline_images,
            "workflow_stage": "article_ready",
        }

    async def write_article_node(state: BlogWorkflowState) -> dict[str, object]:
        written = await write_article_passes(
            request=state["request"],
            plan=state["seo_plan"],
            model=model,
            settings=runtime_settings,
            research=state.get("research_bundle"),
            max_images=max_images,
            image_mode=runtime_settings.image_mode,
            profile=get_profile(state["research_decision"].topic_category),
            content_angle=state["research_decision"].content_angle,
        )
        return {
            "written_article": written,
            "warnings": written.warnings,
        }

    async def generate_inline_images_node(
        state: BlogWorkflowState,
    ) -> dict[str, object]:
        generated = await generate_inline_images(
            state["written_article"].markdown,
            generate_images=state["request"].generate_images,
            max_images=max_images,
            image_provider=image_provider,
            fallback_description=(
                f"Editorial illustration for {state['seo_plan'].title}"
            ),
        )
        return {
            "batched_markdown": generated.markdown,
            "inline_images": generated.inline_images,
            "image_result": generated.image_result,
            "warnings": generated.warnings,
        }

    def prepare_batched_placeholders_node(
        state: BlogWorkflowState,
    ) -> dict[str, object]:
        generated = build_image_placeholders(
            state["written_article"].markdown,
            title=state["seo_plan"].title,
            generate_images=state["request"].generate_images,
            max_images=max_images,
        )
        return {
            "batched_markdown": generated.markdown,
            "image_placeholders": generated.image_placeholders,
            "inline_images": [],
            "image_result": generated.image_result,
            "warnings": generated.warnings,
        }

    def remove_batched_image_markers_node(
        state: BlogWorkflowState,
    ) -> dict[str, object]:
        generated = build_image_placeholders(
            state["written_article"].markdown,
            title=state["seo_plan"].title,
            generate_images=False,
            max_images=max_images,
        )
        return {
            "batched_markdown": generated.markdown,
            "image_placeholders": [],
            "inline_images": [],
            "image_result": generated.image_result,
            "warnings": generated.warnings,
        }

    def assemble_batched_node(state: BlogWorkflowState) -> dict[str, object]:
        written = state["written_article"]
        article: ArticleResult = assemble_batched_article(
            plan=state["seo_plan"],
            markdown=state["batched_markdown"],
            sources=written.sources,
            image_result=state["image_result"],
            inline_images=state.get("inline_images", []),
            image_placeholders=state.get("image_placeholders", []),
            research_mode=state["research_decision"].selected_mode,
            topic_category=state["research_decision"].topic_category,
            content_angle=state["research_decision"].content_angle,
            warnings=state.get("warnings", []),
        )
        article = _apply_fact_guard(
            article,
            request=state["request"],
            plan=state["seo_plan"],
            profile=get_profile(state["research_decision"].topic_category),
            research=state.get("research_bundle"),
            writing_mode=writing_mode,
        )
        return {
            "article": article,
            "image_result": article.image_result,
            "image_placeholders": article.image_placeholders,
            "inline_images": article.inline_images,
            "workflow_stage": "article_ready",
        }

    graph = StateGraph(BlogWorkflowState)
    graph.add_node("assess_research", assess_node)
    graph.add_node("start_research", start_research_node)
    graph.add_node("research", research_node)
    graph.add_node("create_seo_plan", create_plan_node)
    graph.add_edge(START, "assess_research")
    graph.add_conditional_edges(
        "assess_research",
        route_node,
        {
            "model_knowledge": "create_seo_plan",
            "web_research": "start_research",
        },
    )
    graph.add_edge("start_research", "research")
    graph.add_edge("research", "create_seo_plan")

    if write_sections and writing_mode == "batched":
        graph.add_node("write_article", write_article_node)
        graph.add_node("assemble_batched", assemble_batched_node)
        graph.add_edge("create_seo_plan", "write_article")
        if runtime_settings.image_mode == "generate":
            graph.add_node("generate_inline_images", generate_inline_images_node)
            graph.add_edge("write_article", "generate_inline_images")
            graph.add_edge("generate_inline_images", "assemble_batched")
        elif runtime_settings.image_mode == "placeholder":
            graph.add_node(
                "prepare_image_placeholders",
                prepare_batched_placeholders_node,
            )
            graph.add_edge("write_article", "prepare_image_placeholders")
            graph.add_edge("prepare_image_placeholders", "assemble_batched")
        else:
            graph.add_node("remove_image_markers", remove_batched_image_markers_node)
            graph.add_edge("write_article", "remove_image_markers")
            graph.add_edge("remove_image_markers", "assemble_batched")
        graph.add_edge("assemble_batched", END)
    elif write_sections:
        graph.add_node("write_section", write_section_node)
        graph.add_node("collect_sections", collect_sections_node)
        graph.add_node("assemble_article", assemble_article_node)
        graph.add_conditional_edges("create_seo_plan", dispatch_sections)
        graph.add_edge("write_section", "collect_sections")
        if runtime_settings.image_mode == "generate":
            graph.add_node("images", images_node)
            graph.add_edge(["collect_sections", "images"], "assemble_article")
        elif runtime_settings.image_mode == "placeholder":
            graph.add_node(
                "prepare_image_placeholders",
                prepare_section_placeholders_node,
            )
            graph.add_edge("collect_sections", "prepare_image_placeholders")
            graph.add_edge("prepare_image_placeholders", "assemble_article")
        else:
            graph.add_edge("collect_sections", "assemble_article")
        graph.add_edge("assemble_article", END)
    else:
        graph.add_edge("create_seo_plan", END)
    return graph.compile()


def _apply_fact_guard(
    article: ArticleResult,
    *,
    request: BlogRequest,
    plan: SEOPlan,
    profile: TopicProfile,
    research: ResearchBundle | None,
    writing_mode: Literal["batched", "per_section"],
) -> ArticleResult:
    if profile.research_policy == "required" and research is None:
        raise ValueError(
            f"{profile.label} content cannot be completed without web research."
        )
    if research is None or profile.research_policy == "optional":
        return article

    evidence_contexts = []
    if writing_mode == "batched":
        context = compact_research_context(research, profile=profile)
        if context is not None:
            evidence_contexts.append(context)
    else:
        context = compact_research_context(research, profile=profile)
        if context is not None:
            evidence_contexts.append(context)
        for section in plan.sections:
            context = compact_research_context(
                research,
                query_terms=[
                    section.heading,
                    *section.key_points,
                    *section.h3_headings,
                ],
                profile=profile,
            )
            if context is not None:
                evidence_contexts.append(context)

    evidence_texts = [
        item["text"]
        for context in evidence_contexts
        for item in context["evidence"]
    ]
    request_text = " ".join(
        value
        for value in [
            request.topic,
            request.audience or "",
            request.tone,
            request.target_country or "",
            *request.seo_goals,
        ]
        if value
    )
    unsupported = find_unsupported_figures(
        article.markdown,
        evidence_texts,
        request_text,
    )
    if not unsupported:
        return article

    listed_figures = ", ".join(unsupported[:10])
    remaining = len(unsupported) - 10
    suffix = f" (and {remaining} more)" if remaining > 0 else ""
    warning = (
        "These figures in the article are not found in the retrieved research "
        f"and should be verified: {listed_figures}{suffix}."
    )
    warnings = list(article.warnings)
    warnings.append(warning)
    if len(warnings) > 50:
        warnings = warnings[-50:]
    return ArticleResult.model_validate(
        {**article.model_dump(), "warnings": warnings}
    )
