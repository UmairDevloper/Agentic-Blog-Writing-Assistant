import base64
from io import BytesIO
from types import ModuleType

from PIL import Image
from pydantic import SecretStr

from seo_blog_assistant.application import ArticleGenerationError
from seo_blog_assistant.schemas import (
    ArticleResult,
    GeneratedImage,
    ImageGenerationResult,
    ImagePlaceholder,
    InlineImage,
)
from seo_blog_assistant.config import Settings
from seo_blog_assistant.ui import (
    run_ui,
    split_article_markdown,
    split_article_placeholders,
)
from seo_blog_assistant.workflow.image_placeholders import (
    build_placeholder_block,
    strip_image_placeholders,
)


class FakeWorkflow:
    async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
        return {
            "article": ArticleResult(
                title="A practical topic guide",
                meta_description="A useful overview.",
                search_intent="Informational",
                primary_keyword="topic guide",
                related_keyword_ideas=[],
                markdown="# A practical topic guide\n\nA useful guide.\n",
                sources=[],
                image_result=ImageGenerationResult(status="skipped"),
                warnings=["A section was repaired."],
            )
        }


class FakeStreamlit(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.messages: list[tuple[str, str]] = []
        self.submitted = True
        self.selectbox_values: dict[str, str] = {}
        self.selectbox_options: dict[str, list[str]] = {}
        self.groq_api_key = "gsk-personal-test-key"
        self.text_input_options: dict[str, dict[str, object]] = {}
        self.session_state: dict[str, str] = {"groq_api_key": self.groq_api_key}
        self.sidebar = self

    def set_page_config(self, **_kwargs: object) -> None:
        pass

    def title(self, _value: str) -> None:
        pass

    def write(self, _value: str) -> None:
        pass

    def subheader(self, _value: str) -> None:
        pass

    def form(self, _name: str) -> "FakeStreamlit":
        return self

    def cache_resource(self, function):
        return function

    def spinner(self, _label: str) -> "FakeStreamlit":
        return self

    def __enter__(self) -> "FakeStreamlit":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def text_input(
        self,
        label: str,
        *,
        value: str = "",
        max_chars: int | None = None,
        placeholder: str | None = None,
        type: str = "default",
        key: str | None = None,
        help: str | None = None,
    ) -> str:
        self.text_input_options[label] = {
            "type": type,
            "key": key,
            "help": help,
        }
        if label == "Blog topic":
            return "A useful topic"
        if label == "Custom tone":
            return "Warm and witty"
        if label == "Groq API key":
            return self.groq_api_key
        return value

    def selectbox(
        self,
        label: str,
        options: list[str],
        *,
        index: int = 0,
        help: str | None = None,
    ) -> str:
        self.selectbox_options[label] = options
        return self.selectbox_values.get(label, options[index])

    def slider(
        self,
        _label: str,
        *,
        min_value: int,
        max_value: int,
        value: int,
    ) -> int:
        return value

    def checkbox(self, _label: str) -> bool:
        return False

    def form_submit_button(self, _label: str) -> bool:
        return self.submitted

    def markdown(self, value: str, **_kwargs: object) -> None:
        self.messages.append(("markdown", value))

    def error(self, value: str) -> None:
        self.messages.append(("error", value))

    def warning(self, value: str) -> None:
        self.messages.append(("warning", value))

    def success(self, value: str) -> None:
        self.messages.append(("success", value))

    def info(self, value: str) -> None:
        self.messages.append(("info", value))

    def code(self, value: str, *, language: str | None = None) -> None:
        self.messages.append(("code", value))

    def caption(self, value: str) -> None:
        self.messages.append(("caption", value))

    def download_button(self, label: str, **_kwargs: object) -> None:
        self.messages.append(("download", label))

    def button(self, _label: str, *, on_click=None) -> bool:
        return False

    def image(self, _image: object, *, caption: str) -> None:
        self.messages.append(("image", caption))


def test_ui_smoke_renders_article_from_shared_workflow(monkeypatch) -> None:
    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_streamlit)

    run_ui(FakeWorkflow())

    assert ("markdown", "# A practical topic guide\n\nA useful guide.\n") in (
        fake_streamlit.messages
    )
    assert not any(kind == "error" for kind, _ in fake_streamlit.messages)
    assert ("warning", "A section was repaired.") in fake_streamlit.messages


def test_ui_displays_article_generation_error(monkeypatch) -> None:
    class FailedWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            raise ArticleGenerationError(
                "The model output failed validation.",
                status_code=502,
            )

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_streamlit)

    run_ui(FailedWorkflow())

    assert (
        "error",
        "The model output failed validation.",
    ) in fake_streamlit.messages


def test_ui_requires_personal_groq_key_before_using_workflow(monkeypatch) -> None:
    class UnusedWorkflow:
        called = False

        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            self.called = True
            raise AssertionError("workflow must not run without a personal key")

    fake_streamlit = FakeStreamlit()
    fake_streamlit.groq_api_key = ""
    workflow = UnusedWorkflow()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_streamlit)

    run_ui(workflow)

    assert not workflow.called
    assert (
        "error",
        "Enter your Groq API key in the sidebar before creating an article.",
    ) in fake_streamlit.messages
    assert fake_streamlit.text_input_options["Groq API key"]["type"] == "password"


def test_ui_builds_workflow_with_users_key_not_configured_key(monkeypatch) -> None:
    from seo_blog_assistant import ui

    captured_settings: list[Settings] = []

    def fake_build_workflow(*, settings: Settings):
        captured_settings.append(settings)
        return FakeWorkflow()

    fake_streamlit = FakeStreamlit()
    settings = Settings(
        _env_file=None,
        text_model_provider="groq",
        text_model_name="openai/gpt-oss-20b",
        text_model_api_key=SecretStr("owner-configured-key"),
    )
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_streamlit)
    monkeypatch.setattr(ui, "get_settings", lambda: settings)
    monkeypatch.setattr(ui, "build_workflow", fake_build_workflow)

    run_ui()

    assert len(captured_settings) == 1
    assert captured_settings[0].text_model_api_key == SecretStr(
        fake_streamlit.groq_api_key
    )
    assert captured_settings[0].text_model_api_key != settings.text_model_api_key
    assert captured_settings[0].text_model_provider == settings.text_model_provider
    assert captured_settings[0].text_model_name == settings.text_model_name


def test_ui_tone_and_image_dropdowns_create_expected_request(monkeypatch) -> None:
    from seo_blog_assistant import ui

    class CapturingWorkflow:
        request = None

        async def ainvoke(self, input_data: dict[str, object]) -> dict[str, object]:
            self.request = input_data["request"]
            return {
                "article": ArticleResult(
                    title="A practical topic guide",
                    meta_description="A useful overview.",
                    search_intent="Informational",
                    primary_keyword="topic guide",
                    related_keyword_ideas=[],
                    research_mode="web_research",
                    topic_category="sports",
                    content_angle="season preview",
                    markdown="# A practical topic guide\n\nA useful guide.\n",
                    sources=[],
                    image_result=ImageGenerationResult(status="skipped"),
                )
            }

    fake_streamlit = FakeStreamlit()
    fake_streamlit.selectbox_values = {
        "Topic category (optional)": "Sports",
        "Writing tone": "Custom",
        "Images": "Add copyable image prompts",
    }
    workflow = CapturingWorkflow()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_streamlit)
    monkeypatch.setattr(ui, "get_settings", lambda: Settings(_env_file=None))

    run_ui(workflow)

    assert workflow.request.tone == "Warm and witty"
    assert workflow.request.generate_images is True
    assert workflow.request.category == "sports"
    assert "Sports" in fake_streamlit.selectbox_options["Topic category (optional)"]
    assert "Professional" in fake_streamlit.selectbox_options["Writing tone"]
    assert "Custom" in fake_streamlit.selectbox_options["Writing tone"]
    assert "Add copyable image prompts" in fake_streamlit.selectbox_options["Images"]
    assert ("success", "Article ready: A practical topic guide") in (
        fake_streamlit.messages
    )
    assert any(
        kind == "markdown"
        and "LLM-written · Web research assisted" in value
        and "may also draw on existing model knowledge" in value
        for kind, value in fake_streamlit.messages
    )
    assert (
        "caption",
        "Topic category: Sports · season preview",
    ) in fake_streamlit.messages


def test_research_mode_display_names_model_knowledge_and_unknown() -> None:
    from seo_blog_assistant.ui import research_mode_display

    assert research_mode_display("model_knowledge") == (
        "LLM knowledge · No web search",
        "No web search was used. The LLM wrote this draft from its existing "
        "knowledge.",
    )
    assert research_mode_display("web_research") == (
        "LLM-written · Web research assisted",
        "The LLM wrote this article with retrieved web sources as context; "
        "it may also draw on existing model knowledge.",
    )
    assert research_mode_display(None)[0] == "Research source not reported"


def test_ui_displays_inline_image_provider_errors(monkeypatch) -> None:
    class FailedInlineWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            return {
                "article": ArticleResult(
                    title="A practical topic guide",
                    meta_description="A useful overview.",
                    search_intent="Informational",
                    primary_keyword="topic guide",
                    related_keyword_ideas=[],
                    markdown="# A practical topic guide\n\nA useful guide.\n",
                    sources=[],
                    image_result=ImageGenerationResult(
                        status="failed",
                        message="All image generations failed.",
                    ),
                    inline_images=[
                        InlineImage(
                            image_id="img_1",
                            after_heading="Energy options",
                            alt_text="Energy options",
                            status="failed",
                            message="Hugging Face monthly credits exhausted",
                        )
                    ],
                )
            }

    from seo_blog_assistant import ui

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_streamlit)
    monkeypatch.setattr(ui, "get_settings", lambda: Settings(_env_file=None))

    run_ui(FailedInlineWorkflow())

    assert (
        "error",
        "Image img_1 failed: Hugging Face monthly credits exhausted",
    ) in fake_streamlit.messages


def test_split_article_markdown_keeps_inline_image_positions() -> None:
    markdown = (
        "# Article\n\nBefore image.\n\n"
        "![A generated image](image:img_1)\n\nAfter image."
    )

    segments = split_article_markdown(markdown)

    assert segments == [
        ("# Article\n\nBefore image.\n\n", None),
        ("", "img_1"),
        ("\n\nAfter image.", None),
    ]


def test_split_article_placeholders_rejoins_markdown_in_document_order() -> None:
    block = build_placeholder_block(
        "img_1",
        "A calm illustrated forest at sunrise, soft colors, wide composition.",
        "A calm forest",
    )
    markdown = f"# Article\n\n## Forest\n{block}Useful text.\n"

    segments = split_article_placeholders(markdown)

    assert "".join(segment for segment, _image_id in segments) == markdown
    assert [image_id for _segment, image_id in segments if image_id] == ["img_1"]
    assert strip_image_placeholders(markdown) == "# Article\n\n## Forest\nUseful text.\n"


def test_ui_renders_placeholder_prompt_and_downloads_both_versions(monkeypatch) -> None:
    prompt = (
        "A calm illustrated forest at sunrise, soft colors, wide composition, "
        "quiet atmosphere, layered trees, warm light, open foreground."
    )
    block = build_placeholder_block("img_1", prompt, prompt)
    markdown = f"# A practical topic guide\n\n## Section 1{block}Article text.\n"

    class PlaceholderWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            return {
                "article": ArticleResult(
                    title="A practical topic guide",
                    meta_description="A useful overview.",
                    search_intent="Informational",
                    primary_keyword="topic guide",
                    related_keyword_ideas=[],
                    markdown=markdown,
                    sources=[],
                    image_result=ImageGenerationResult(status="skipped"),
                    image_placeholders=[
                        ImagePlaceholder(
                            image_id="img_1",
                            after_heading="Section 1",
                            prompt=prompt,
                            alt_text=prompt[:120],
                        )
                    ],
                    warnings=["A fallback prompt was created."],
                )
            }

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_streamlit)

    run_ui(PlaceholderWorkflow())

    assert ("info", "Place an image here (img_1)") in fake_streamlit.messages
    assert ("code", prompt) in fake_streamlit.messages
    assert any(
        kind == "caption" and value.startswith("Alt text:")
        for kind, value in fake_streamlit.messages
    )
    assert (
        "download",
        "Download Markdown with placeholders",
    ) in fake_streamlit.messages
    assert (
        "download",
        "Download Markdown without placeholders",
    ) in fake_streamlit.messages
    assert ("warning", "A fallback prompt was created.") in fake_streamlit.messages


def test_ui_renders_inline_images_between_markdown_segments(monkeypatch) -> None:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color="blue").save(buffer, format="PNG")
    image = GeneratedImage(
        model_name="owner/open-image-model",
        prompt="Editorial illustration.",
        alt_text="A generated image",
        image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )
    markdown = (
        "# A practical topic guide\n\nBefore image.\n\n"
        "![A generated image](image:img_1)\n\nAfter image."
    )

    class InlineImageWorkflow:
        async def ainvoke(self, _input: dict[str, object]) -> dict[str, object]:
            return {
                "article": ArticleResult(
                    title="A practical topic guide",
                    meta_description="A useful overview.",
                    search_intent="Informational",
                    primary_keyword="topic guide",
                    related_keyword_ideas=[],
                    markdown=markdown,
                    sources=[],
                    image_result=ImageGenerationResult(
                        status="completed",
                        image=image,
                    ),
                    inline_images=[
                        InlineImage(
                            image_id="img_1",
                            after_heading="A practical topic guide",
                            alt_text=image.alt_text,
                            status="completed",
                            image=image,
                        )
                    ],
                )
            }

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_streamlit)

    run_ui(InlineImageWorkflow())

    content_events = [
        kind
        for kind, _value in fake_streamlit.messages
        if kind in {"markdown", "image"}
    ]
    assert content_events[-3:] == ["markdown", "image", "markdown"]
    assert ("image", "A generated image") in fake_streamlit.messages
