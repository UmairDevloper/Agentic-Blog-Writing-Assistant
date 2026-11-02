import asyncio
import json
from typing import TypeVar

from pydantic import BaseModel

from seo_blog_assistant.providers.text_model import TextResult
from seo_blog_assistant.schemas import BlogRequest, ResearchDecision, SEOPlan
from seo_blog_assistant.workflow.planning import create_seo_plan
from seo_blog_assistant.workflow.topic_profiles import (
    TOPIC_PROFILES,
    TopicCategory,
    get_profile,
    profile_prompt_block,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def test_profiles_cover_all_categories_and_unknown_values_use_general() -> None:
    expected: set[TopicCategory] = {
        "technology",
        "sports",
        "food",
        "politics",
        "business_finance",
        "health_science",
        "travel_culture",
        "entertainment",
        "history_education",
        "general",
    }

    assert set(TOPIC_PROFILES) == expected
    assert get_profile(None) == TOPIC_PROFILES["general"]
    assert get_profile("unknown") == TOPIC_PROFILES["general"]


def test_profile_prompt_blocks_are_short_and_contain_evidence_rules() -> None:
    for profile in TOPIC_PROFILES.values():
        block = profile_prompt_block(profile)
        assert len(profile.outline_elements) in range(4, 8)
        assert len(profile.search_hints) <= 2
        assert len(block) <= 900
        assert profile.evidence_rules in block
        assert profile.style_notes in block


class OutlineModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(
        self,
        *,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
    ) -> SchemaT:
        self.prompts.append(f"{system_prompt}\n{user_prompt}")
        payload = json.loads(user_prompt)
        category = payload["request"]["category"]
        headings = {
            "sports": [
                "Recent results and standings",
                "Key stats and records",
                "Past seasons",
            ],
            "food": [
                "Origins and culture",
                "Ingredients and flavor",
                "Preparation techniques",
            ],
        }[category]
        plan = SEOPlan(
            title=f"A guide to {category}",
            meta_description="A useful topic overview.",
            search_intent="Informational",
            primary_keyword=category,
            related_keyword_ideas=[],
            target_words=1200,
            introduction_brief="Introduce the topic.",
            introduction_word_budget=100,
            sections=[
                {
                    "id": f"section_{index}",
                    "position": index,
                    "heading": heading,
                    "purpose": "Explain a reader-relevant part of the topic.",
                    "key_points": ["Explain one important idea"],
                    "h3_headings": ["What readers should know"],
                    "word_budget": 300,
                }
                for index, heading in enumerate(headings)
            ],
            conclusion_brief="Summarize the main points.",
            conclusion_word_budget=100,
        )
        return schema.model_validate(plan)

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> TextResult:
        raise AssertionError("This test only calls the planner.")


def test_category_profiles_produce_different_outlines_without_extra_calls() -> None:
    model = OutlineModel()
    plans: dict[str, SEOPlan] = {}
    for category in ("sports", "food"):
        decision = ResearchDecision(
            selected_mode="model_knowledge",
            rationale="General knowledge is sufficient.",
            estimated_knowledge_sufficient=True,
            confidence=0.8,
            topic_category=category,
            content_angle=f"{category} overview",
        )
        plans[category] = asyncio.run(
            create_seo_plan(
                BlogRequest(topic=f"A {category} topic", category=category),
                decision,
                model,
            )
        )

    sports_headings = [section.heading for section in plans["sports"].sections]
    food_headings = [section.heading for section in plans["food"].sections]
    assert sports_headings != food_headings
    assert any("stats" in heading.casefold() for heading in sports_headings)
    assert any("ingredients" in heading.casefold() for heading in food_headings)
    assert len(model.prompts) == 2
    assert "Category: Sports" in model.prompts[0]
    assert "Category: Food" not in model.prompts[0]
    assert "Category: Food" in model.prompts[1]
    assert "Category: Sports" not in model.prompts[1]
    assert "Do not force problem, limitations, or solutions" in model.prompts[1]
