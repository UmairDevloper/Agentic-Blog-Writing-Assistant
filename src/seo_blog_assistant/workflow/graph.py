"""LangGraph construction for research, planning, and article writing."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from seo_blog_assistant.providers.image import ImageProvider, ImageProviderError
from seo_blog_assistant.providers.search import (
    ResearchProvider,
    ResearchProviderError,
)
from seo_blog_assistant.providers.text_model import StructuredTextModel
from seo_blog_assistant.schemas import (
    ArticleBookends,
    ArticleResult,
    ImageGenerationResult,
    ResearchMode,
    SectionDraft,
)
from seo_blog_assistant.workflow.assembly import assemble_article, write_bookends
from seo_blog_assistant.workflow.planning import create_seo_plan
from seo_blog_assistant.workflow.research import assess_research, research_route
from seo_blog_assistant.workflow.state import BlogWorkflowState
from seo_blog_assistant.workflow.writing import write_section


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
) -> CompiledStateGraph[
    BlogWorkflowState,
    None,
    BlogWorkflowState,
    BlogWorkflowState,
]:
    """Build the research, planning, and concurrent section-writing workflow."""
    return _build_graph(
        model,
        search_provider,
        image_provider,
        write_sections=True,
    )


def _build_graph(
    model: StructuredTextModel,
    search_provider: ResearchProvider | None,
    image_provider: ImageProvider | None = None,
    *,
    write_sections: bool,
) -> CompiledStateGraph[
    BlogWorkflowState,
    None,
    BlogWorkflowState,
    BlogWorkflowState,
]:
    async def assess_node(state: BlogWorkflowState) -> dict[str, object]:
        decision = await assess_research(state["request"], model)
        return {"research_decision": decision}

    def route_node(state: BlogWorkflowState) -> ResearchMode:
        route = research_route(state["research_decision"])
        if route == "web_research" and search_provider is None:
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
        bundle = await search_provider.search_and_retrieve(state["request"])
        return {"research_bundle": bundle, "workflow_stage": "research_completed"}

    async def create_plan_node(state: BlogWorkflowState) -> dict[str, object]:
        plan = await create_seo_plan(
            request=state["request"],
            decision=state["research_decision"],
            model=model,
            research=state.get("research_bundle"),
        )
        return {"seo_plan": plan, "workflow_stage": "plan_ready"}

    def dispatch_sections(state: BlogWorkflowState) -> list[Send]:
        return [
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

    async def write_section_node(state: BlogWorkflowState) -> dict[str, object]:
        draft = await write_section(
            request=state["request"],
            plan=state["seo_plan"],
            section=state["active_section"],
            model=model,
            research=state.get("research_bundle"),
        )
        return {"section_drafts": [draft]}

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
        return {
            "ordered_section_drafts": sorted(drafts, key=lambda draft: draft.position)
        }

    def route_images(state: BlogWorkflowState) -> str:
        return "generate_images" if state["request"].generate_images else "skip_images"

    async def generate_images_node(state: BlogWorkflowState) -> dict[str, object]:
        if image_provider is None:
            result = ImageGenerationResult(
                status="failed",
                message=(
                    "Image generation was requested, but no image provider is configured."
                ),
            )
        else:
            plan = state["seo_plan"]
            prompt = (
                "Create a clear, professional editorial illustration for a blog article "
                f"about {state['request'].topic!r}, titled {plan.title!r}. "
                "Use a clean composition, no text, logos, or watermarks."
            )
            try:
                image = await image_provider.generate_image(
                    prompt,
                    f"Illustration for {plan.title}",
                )
            except ImageProviderError:
                result = ImageGenerationResult(
                    status="failed",
                    message=(
                        "Image generation failed; the article draft is available "
                        "without generated images."
                    ),
                )
            else:
                result = ImageGenerationResult(status="completed", image=image)
        return {
            "image_result": result,
            "workflow_stage": f"images_{result.status}",
        }

    async def skip_images_node(
        _state: BlogWorkflowState,
    ) -> dict[str, object]:
        result = ImageGenerationResult(
            status="skipped",
            message="Image generation was not requested.",
        )
        return {"image_result": result, "workflow_stage": "images_skipped"}

    async def assemble_article_node(
        state: BlogWorkflowState,
    ) -> dict[str, object]:
        bookends: ArticleBookends = await write_bookends(
            request=state["request"],
            plan=state["seo_plan"],
            model=model,
            research=state.get("research_bundle"),
        )
        article: ArticleResult = assemble_article(
            plan=state["seo_plan"],
            bookends=bookends,
            section_drafts=state["ordered_section_drafts"],
            research=state.get("research_bundle"),
            image_result=state["image_result"],
        )
        return {
            "bookends": bookends,
            "article": article,
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

    if write_sections:
        graph.add_node("write_section", write_section_node)
        graph.add_node("collect_sections", collect_sections_node)
        graph.add_node("generate_images", generate_images_node)
        graph.add_node("skip_images", skip_images_node)
        graph.add_node("assemble_article", assemble_article_node)
        graph.add_conditional_edges("create_seo_plan", dispatch_sections)
        graph.add_edge("write_section", "collect_sections")
        graph.add_conditional_edges(
            "collect_sections",
            route_images,
            {
                "generate_images": "generate_images",
                "skip_images": "skip_images",
            },
        )
        graph.add_edge("generate_images", "assemble_article")
        graph.add_edge("skip_images", "assemble_article")
        graph.add_edge("assemble_article", END)
    else:
        graph.add_edge("create_seo_plan", END)
    return graph.compile()
