"""Streamlit entry point for the shared blog-generation workflow."""

import asyncio
import base64
import html
import logging
import re
from io import BytesIO

from pydantic import SecretStr, ValidationError

from seo_blog_assistant.application import (
    ArticleGenerationError,
    BlogWorkflow,
    WorkflowConfigurationError,
    build_workflow,
    generate_article,
)
from seo_blog_assistant.config import Settings, get_settings
from seo_blog_assistant.logging_config import configure_logging
from seo_blog_assistant.providers.search import ResearchProviderError
from seo_blog_assistant.providers.text_model import TextModelError
from seo_blog_assistant.schemas import BlogRequest
from seo_blog_assistant.workflow.image_placeholders import (
    split_placeholder_blocks,
    strip_image_placeholders,
)
from seo_blog_assistant.workflow.topic_profiles import TOPIC_PROFILES, get_profile

logger = logging.getLogger(__name__)
_INLINE_IMAGE_PATTERN = re.compile(r"!\[(?:\\.|[^\]])*\]\(image:(img_[1-4])\)")


def split_article_markdown(markdown: str) -> list[tuple[str, str | None]]:
    """Split Markdown around generated-image placeholders in display order."""
    segments: list[tuple[str, str | None]] = []
    position = 0
    for match in _INLINE_IMAGE_PATTERN.finditer(markdown):
        segments.append((markdown[position : match.start()], None))
        segments.append(("", match.group(1)))
        position = match.end()
    segments.append((markdown[position:], None))
    return segments


def split_article_placeholders(markdown: str) -> list[tuple[str, str | None]]:
    """Split Markdown around copyable image-prompt placeholder blocks."""
    return split_placeholder_blocks(markdown)


def research_mode_display(research_mode: str | None) -> tuple[str, str]:
    """Explain whether completed research supplemented LLM-written prose."""
    if research_mode == "web_research":
        return (
            "LLM-written · Web research assisted",
            "The LLM wrote this article with retrieved web sources as context; "
            "it may also draw on existing model knowledge.",
        )
    if research_mode == "model_knowledge":
        return (
            "LLM knowledge · No web search",
            "No web search was used. The LLM wrote this draft from its existing "
            "knowledge.",
        )
    return (
        "Research source not reported",
        "This workflow did not report whether web research was completed.",
    )


def _settings_with_user_groq_key(settings: Settings, api_key: str) -> Settings:
    """Return request-local settings with the UI user's key replacing any default."""
    return settings.model_copy(
        update={"text_model_api_key": SecretStr(api_key.strip())}
    )


def run_ui(workflow: BlogWorkflow | None = None) -> None:
    """Render a topic-to-article interface using the shared application service."""
    import streamlit as st

    settings = get_settings()
    configure_logging(settings.log_level)
    st.set_page_config(
        page_title="SEO Blog Writing Assistant",
        page_icon="✍️",
        layout="wide",
    )
    st.sidebar.subheader("Your Groq API key")
    groq_api_key = st.sidebar.text_input(
        "Groq API key",
        type="password",
        key="groq_api_key",
        placeholder="Enter your personal Groq API key",
        help="Used to build this request's workflow. It overrides any configured "
        "key and is not saved to project configuration.",
    )
    st.sidebar.caption(
        "Your key is used by this app's server to call Groq. Avoid shared or "
        "untrusted deployments, and clear the field when finished."
    )
    st.sidebar.button(
        "Clear API key",
        on_click=lambda: st.session_state.update({"groq_api_key": ""}),
    )
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(ellipse at 12% 0%, rgba(119, 89, 255, .16), transparent 34%),
                radial-gradient(ellipse at 88% 12%, rgba(39, 197, 180, .10), transparent 30%),
                #f5f7fb;
            color: #20243a;
        }
        [data-testid="stHeader"] { background: rgba(245, 247, 251, .82); }
        .block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 4rem; }
        .brand-hero {
            position: relative; overflow: hidden; color: white;
            padding: 2.35rem 2.6rem; margin: .25rem 0 1.65rem;
            border-radius: 24px;
            background: linear-gradient(118deg, #171b38 0%, #35316c 60%, #5260a9 100%);
            box-shadow: 0 22px 55px rgba(29, 34, 76, .19);
        }
        .brand-hero:after {
            content: ""; position: absolute; width: 250px; height: 250px;
            right: -48px; top: -105px; border-radius: 50%;
            border: 1px solid rgba(255,255,255,.18);
            box-shadow: 0 0 0 28px rgba(255,255,255,.035), 0 0 0 58px rgba(255,255,255,.025);
        }
        .brand-kicker { color: #bcb8ff; font-size: .76rem; font-weight: 750;
            letter-spacing: .16em; text-transform: uppercase; }
        .brand-title { color: white; margin: .48rem 0 .45rem; font-size: clamp(2rem, 4vw, 3.15rem);
            letter-spacing: -.045em; line-height: 1.08; font-weight: 780; }
        .brand-copy { color: #d7daf1; font-size: 1.05rem; max-width: 690px; margin: 0; }
        .workflow-chip {
            display: inline-block; margin-top: 1.1rem; padding: .38rem .72rem;
            border: 1px solid rgba(255,255,255,.2); border-radius: 999px;
            color: #e4e6ff; font-size: .78rem; background: rgba(255,255,255,.08);
        }
        [data-testid="stForm"] {
            padding: 1.35rem 1.55rem 1.15rem; background: rgba(255,255,255,.92);
            border: 1px solid #e6e9f2; border-radius: 20px;
            box-shadow: 0 12px 36px rgba(34, 44, 80, .07);
        }
        [data-testid="stForm"] [data-testid="stMarkdownContainer"] h3 {
            color: #252b48 !important; font-size: 1.08rem; margin: .45rem 0 .9rem;
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stTextInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label,
        [data-testid="stCheckbox"] label {
            color: #303650 !important; font-weight: 620;
        }
        [data-testid="stTextInput"] [data-baseweb="input"],
        [data-testid="stTextArea"] [data-baseweb="textarea"],
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            color: #20243a !important;
            background: #fafbff !important;
            border: 1px solid #cbd1e0 !important;
            border-radius: 10px !important;
            box-shadow: 0 1px 2px rgba(32, 36, 58, .04);
        }
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextArea"] [data-testid="stTextAreaRootElement"],
        [data-testid="stSelectbox"] [role="group"] {
            color: #20243a !important;
            background: #fafbff !important;
            border: 1px solid #cbd1e0 !important;
            border-radius: 10px !important;
            box-shadow: 0 1px 2px rgba(32, 36, 58, .04);
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] [role="combobox"] {
            color: #20243a !important;
            background: transparent !important;
            caret-color: #5146c8;
        }
        [data-testid="stTextInputField"] {
            color: #20243a !important;
            background: transparent !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color: #707790 !important;
            opacity: 1;
        }
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"] {
            color: #20243a !important;
            background: #fff !important;
        }
        [role="option"] {
            color: #303650 !important;
            background: #fff !important;
        }
        [role="option"]:hover,
        [aria-selected="true"][role="option"] {
            color: #302773 !important;
            background: #efedff !important;
        }
        [data-testid="stCheckbox"] label {
            align-items: center;
        }
        [data-testid="stCheckbox"] input {
            accent-color: #6156d9;
        }
        [data-testid="stSlider"] [role="slider"] {
            background: #6156d9;
            border-color: #6156d9;
        }
        [data-testid="stFormSubmitButton"] button {
            min-height: 3.05rem; border: 0; border-radius: 11px;
            color: white; font-weight: 720; letter-spacing: .01em;
            background: linear-gradient(100deg, #5545c8, #7664e8);
            box-shadow: 0 8px 18px rgba(85, 69, 200, .22);
        }
        [data-testid="stFormSubmitButton"] button:hover {
            border: 0; color: white; filter: brightness(1.07);
            box-shadow: 0 11px 22px rgba(85, 69, 200, .3);
        }
        .research-card {
            display: flex; align-items: flex-start; gap: .85rem;
            padding: 1rem 1.15rem; margin: .5rem 0 1rem;
            border: 1px solid #e5e8f1; border-radius: 14px; background: white;
        }
        .research-mark {
            flex: 0 0 auto; display: grid; place-items: center; width: 2.25rem; height: 2.25rem;
            border-radius: 10px; background: #eeecff; color: #5143bd; font-size: 1.05rem;
        }
        .research-title { color: #242a43; font-weight: 730; margin: 0 0 .15rem; }
        .research-copy { color: #68708a; font-size: .9rem; margin: 0; }
        .stDownloadButton button { border-radius: 9px; font-weight: 650; }
        [data-testid="stAlert"] { border-radius: 12px; }
        @media (max-width: 640px) {
            .brand-hero { padding: 1.65rem 1.3rem; border-radius: 18px; }
            [data-testid="stForm"] { padding: 1rem; }
            .block-container { padding: 1rem .75rem 3rem; }
        }
        </style>
        <section class="brand-hero">
          <div class="brand-kicker">Research-led writing studio</div>
          <div class="brand-title">SEO Blog Writing Assistant</div>
          <p class="brand-copy">Turn a clear idea into a polished, search-ready article — with the right research, structure, and image prompts.</p>
          <span class="workflow-chip">LangGraph workflow&nbsp; · &nbsp;Validated article output</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    tone_options = [
        "Professional",
        "Conversational",
        "Friendly",
        "Authoritative",
        "Educational",
        "Persuasive",
        "Custom",
    ]
    category_options = ["Auto-detect"] + [
        profile.label
        for profile in TOPIC_PROFILES.values()
        if profile.category != "general"
    ]
    image_options = (
        ["No images", "Add copyable image prompts"]
        if settings.image_mode == "placeholder"
        else ["No images", "Generate images with Hugging Face"]
        if settings.image_mode == "generate"
        else ["No images"]
    )
    with st.form("article_request"):
        st.subheader("1. Describe your article")
        topic = st.text_input(
            "Blog topic",
            max_chars=300,
            placeholder="For example: Practical ways to save energy at home",
        )
        audience = st.text_input("Target audience (optional)", max_chars=200)
        country = st.text_input("Target country (optional)", max_chars=80)
        category_choice = st.selectbox(
            "Topic category (optional)",
            options=category_options,
            index=0,
            help="Auto-detect uses the topic; choosing a category overrides it.",
        )
        selected_category = next(
            (
                profile.category
                for profile in TOPIC_PROFILES.values()
                if profile.label == category_choice
            ),
            None,
        )
        tone_choice = st.selectbox(
            "Writing tone",
            options=tone_options,
            index=0,
            help="Choose a style or select Custom to enter your own.",
        )
        tone = (
            st.text_input("Custom tone", max_chars=80, placeholder="Warm and witty")
            if tone_choice == "Custom"
            else tone_choice.lower()
        )
        st.subheader("2. Set article preferences")
        target_words = st.slider(
            "Target words",
            min_value=300,
            max_value=5000,
            value=1200,
        )
        require_freshness = st.checkbox("Require current information")
        require_citations = st.checkbox("Require source citations")
        image_choice = st.selectbox(
            "Images",
            options=image_options,
            index=0,
            help=(
                "The default placeholder mode adds copyable prompts without "
                "using image credits. Set SEO_BLOG_IMAGE_MODE=generate to call "
                "the configured Hugging Face image provider."
            ),
        )
        generate_images = image_choice != "No images"
        submitted = st.form_submit_button("✨ Create my article")

    if not submitted:
        return

    if not groq_api_key.strip():
        st.error("Enter your Groq API key in the sidebar before creating an article.")
        return

    try:
        request = BlogRequest(
            topic=topic,
            audience=audience or None,
            category=selected_category,
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
        active_workflow = (
            workflow
            if workflow is not None
            else build_workflow(
                settings=_settings_with_user_groq_key(settings, groq_api_key)
            )
        )
        with st.spinner("Generating article... this can take a minute"):
            article = asyncio.run(generate_article(request, active_workflow))
    except (
        WorkflowConfigurationError,
        ResearchProviderError,
        TextModelError,
        ArticleGenerationError,
    ) as exc:
        logger.warning("Article generation failed failure_type=%s", type(exc).__name__)
        st.error(str(exc))
        return

    research_title, research_copy = research_mode_display(article.research_mode)
    research_icon = "⌕" if article.research_mode == "web_research" else "✦"
    st.markdown(
        f"""
        <div class="research-card">
          <span class="research-mark">{research_icon}</span>
          <div>
            <p class="research-title">{html.escape(research_title)}</p>
            <p class="research-copy">{html.escape(research_copy)}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.success(f"Article ready: {article.title}")
    category_profile = get_profile(article.topic_category)
    angle_copy = f" · {article.content_angle}" if article.content_angle else ""
    st.caption(f"Topic category: {category_profile.label}{angle_copy}")
    image_count = len(article.image_placeholders or article.inline_images)
    st.caption(
        f"{len(strip_image_placeholders(article.markdown).split())} words"
        f" | {image_count} image slot(s)"
    )

    if article.image_placeholders:
        placeholders = {
            placeholder.image_id: placeholder
            for placeholder in article.image_placeholders
        }
        for segment, image_id in split_article_placeholders(article.markdown):
            if image_id is None:
                if segment:
                    st.markdown(segment)
                continue
            placeholder = placeholders.get(image_id)
            if placeholder is None:
                st.error(f"Article references missing image prompt {image_id}.")
                continue
            st.info(f"Place an image here ({image_id})")
            st.code(placeholder.prompt, language=None)
            st.caption(f"Alt text: {placeholder.alt_text}")
    else:
        inline_images = {image.image_id: image for image in article.inline_images}
        for text_segment, image_id in split_article_markdown(article.markdown):
            if text_segment:
                st.markdown(text_segment)
            if image_id is None:
                continue
            image = inline_images.get(image_id)
            if image is None:
                st.error(f"Article references missing inline image {image_id}.")
            elif image.status == "completed" and image.image is not None:
                image_data = base64.b64decode(
                    image.image.image_base64,
                    validate=True,
                )
                st.image(BytesIO(image_data), caption=image.alt_text)
            else:
                st.warning(image.message or "This inline image could not be generated.")

    st.download_button(
        "Download Markdown with placeholders",
        data=article.markdown,
        file_name="article.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download Markdown without placeholders",
        data=strip_image_placeholders(article.markdown),
        file_name="article.md",
        mime="text/markdown",
    )

    for warning in article.warnings:
        st.warning(warning)
    failed_inline_images = [
        image for image in article.inline_images if image.status == "failed"
    ]
    for image in failed_inline_images:
        st.error(f"Image {image.image_id} failed: {image.message}")
    if (
        not article.inline_images
        and article.image_result.status == "completed"
        and article.image_result.image
    ):
        image_data = base64.b64decode(
            article.image_result.image.image_base64,
            validate=True,
        )
        st.image(
            BytesIO(image_data),
            caption=article.image_result.image.alt_text,
        )
    elif article.image_result.status == "failed" and not failed_inline_images:
        st.error(article.image_result.message)


if __name__ == "__main__":
    run_ui()
