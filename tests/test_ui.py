from types import ModuleType

from seo_blog_assistant.schemas import ArticleResult, ImageGenerationResult
from seo_blog_assistant.ui import run_ui


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
            )
        }


class FakeStreamlit(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.messages: list[tuple[str, str]] = []
        self.submitted = True

    def set_page_config(self, **_kwargs: object) -> None:
        pass

    def title(self, _value: str) -> None:
        pass

    def write(self, _value: str) -> None:
        pass

    def form(self, _name: str) -> "FakeStreamlit":
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
    ) -> str:
        return "A useful topic" if label == "Blog topic" else value

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

    def markdown(self, value: str) -> None:
        self.messages.append(("markdown", value))

    def error(self, value: str) -> None:
        self.messages.append(("error", value))

    def warning(self, value: str) -> None:
        self.messages.append(("warning", value))


def test_ui_smoke_renders_article_from_shared_workflow(monkeypatch) -> None:
    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_streamlit)

    run_ui(FakeWorkflow())

    assert ("markdown", "# A practical topic guide\n\nA useful guide.\n") in (
        fake_streamlit.messages
    )
    assert not any(kind == "error" for kind, _ in fake_streamlit.messages)
