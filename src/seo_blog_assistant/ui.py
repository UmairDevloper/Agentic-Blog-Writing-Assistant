"""Streamlit entry point for the shared blog-generation workflow."""

import asyncio
import base64
from io import BytesIO

from pydantic import ValidationError

from seo_blog_assistant.application import (
    BlogWorkflow,
    WorkflowConfigurationError,
    build_workflow,
    generate_article,
)
from seo_blog_assistant.providers.search import ResearchProviderError
from seo_blog_assistant.providers.text_model import TextModelError
from seo_blog_assistant.schemas import BlogRequest


def run_ui(workflow: BlogWorkflow | None = None) -> None:
    """Render a topic-to-article interface using the shared application service."""
    import streamlit as st

    st.set_page_config(page_title="SEO Blog Writing Assistant")
    st.title("SEO Blog Writing Assistant")
    st.write("Create a structured, SEO-focused article from a topic.")

    with st.form("article_request"):
        topic = st.text_input("Blog topic", max_chars=300)
        audience = st.text_input("Target audience (optional)", max_chars=200)
        country = st.text_input("Target country (optional)", max_chars=80)
        tone = st.text_input("Tone", value="professional", max_chars=80)
        target_words = st.slider(
            "Target words",
            min_value=300,
            max_value=5000,
            value=1200,
        )
        require_freshness = st.checkbox("Require current information")
        require_citations = st.checkbox("Require source citations")
        generate_images = st.checkbox("Generate an illustrative image")
        submitted = st.form_submit_button("Generate article")

    if not submitted:
        return

    try:
        request = BlogRequest(
            topic=topic,
            audience=audience or None,
            target_country=country or None,
            tone=tone,
            target_words=target_words,
            require_freshness=require_freshness,
            require_citations=require_citations,
            generate_images=generate_images,
        )
    except ValidationError as exc:
        st.error(f"Check the request fields: {exc}")
        return

    try:
        active_workflow = workflow or build_workflow()
        article = asyncio.run(generate_article(request, active_workflow))
    except (
        WorkflowConfigurationError,
        ResearchProviderError,
        TextModelError,
    ) as exc:
        st.error(str(exc))
        return

    st.markdown(article.markdown)
    if article.image_result.status == "completed" and article.image_result.image:
        image_data = base64.b64decode(
            article.image_result.image.image_base64,
            validate=True,
        )
        st.image(
            BytesIO(image_data),
            caption=article.image_result.image.alt_text,
        )
    elif article.image_result.status == "failed":
        st.warning(article.image_result.message)


if __name__ == "__main__":
    run_ui()
