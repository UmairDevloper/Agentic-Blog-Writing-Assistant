"""Pure checks for figures in research-dependent article drafts."""

import re

from seo_blog_assistant.workflow.image_placeholders import strip_image_placeholders

_CODE_BLOCK = re.compile(r"(?s)(```.*?```|~~~.*?~~~)")
_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_FIGURE = re.compile(
    rf"(?<![\w])(?P<figure>(?:[$€£]\s*)?{_NUMBER}"
    r"(?:\s*(?:thousand|million|billion|trillion))?(?:\s*%)?)(?!\w)",
    re.IGNORECASE,
)
_LIST_NUMBER = re.compile(r"^\s*\d+[.)]\s+")
_HEADING_NUMBER = re.compile(r"^(\s*#{1,6}\s+)\d+[.)]?\s+")
_WORD_BUDGET = re.compile(r"(?i)^\s*-?\s*words?(?:\s+count)?\b")
_NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def find_unsupported_figures(
    markdown: str,
    evidence_texts: list[str],
    request_text: str = "",
) -> list[str]:
    """Return numeric claims absent from the research the writer actually saw."""
    article_text = strip_image_placeholders(markdown)
    article_text = _CODE_BLOCK.sub(" ", article_text)
    article_text = _URL.sub(" ", article_text)
    article_text = "\n".join(
        _without_structural_numbers(line) for line in article_text.splitlines()
    )

    reference_text = " ".join([*evidence_texts, request_text])
    normalized_reference = _normalize_text(reference_text)
    found: list[str] = []
    seen: set[str] = set()

    for match in _FIGURE.finditer(article_text):
        figure = match.group("figure").strip()
        if _is_word_budget(article_text, match.end()):
            continue
        normalized_figure = _normalize_text(figure)
        if normalized_figure in seen:
            continue
        if _contains_figure(normalized_reference, normalized_figure):
            continue
        seen.add(normalized_figure)
        found.append(figure)
    return found


def _without_structural_numbers(line: str) -> str:
    """Remove only heading/list ordinals, retaining figures in the prose."""
    line = _HEADING_NUMBER.sub(r"\1", line)
    return _LIST_NUMBER.sub("", line)


def _is_word_budget(text: str, end: int) -> bool:
    return _WORD_BUDGET.match(text[end : end + 30]) is not None


def _normalize_text(value: str) -> str:
    normalized = value.casefold().replace(",", "").replace("$", "")
    normalized = normalized.replace("€", "").replace("£", "")

    def normalize_number(match: re.Match[str]) -> str:
        number = match.group(0)
        if "." not in number:
            return number
        return number.rstrip("0").rstrip(".")

    normalized = _NUMBER_TOKEN.sub(normalize_number, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_figure(reference_text: str, figure: str) -> bool:
    escaped = re.escape(figure)
    return re.search(rf"(?<!\d){escaped}(?!\d)", reference_text) is not None
