# Project Scope and Implementation Notes

## Purpose

The SEO Blog Writing Assistant turns a topic and article preferences into a
validated, SEO-oriented Markdown article. It supports optional web research,
optional image prompts or hosted image generation, a Streamlit interface, and a
FastAPI API. The existing prototype material under `backend/` is preserved and
is not part of this application.

## Technology used

- Python 3.12 with the standard-library `venv` workflow and `requirements.txt`
- Pydantic v2 for settings, request validation, and article/data schemas
- LangChain and Groq-hosted text inference for structured planning and Markdown
  writing
- LangGraph for research routing, writing, optional image processing, and
  assembly
- Tavily for optional authenticated web search and retrieved source evidence
- Hugging Face Inference Providers for optional hosted image generation
- FastAPI for HTTP requests and background article jobs
- Streamlit for the interactive user interface
- pytest with fake providers for offline automated tests

Provider credentials are read from environment-backed settings or, for Groq in
the Streamlit UI, from the user's password-masked sidebar field. Do not put
actual credentials in this document or commit them.

## How it is implemented

### Shared application and workflow

The API and UI both call the shared application service in
`src/seo_blog_assistant/application.py`. Provider-specific code is isolated in
`src/seo_blog_assistant/providers/`, while workflow nodes and graph routing
live under `src/seo_blog_assistant/workflow/`. Pydantic schemas in
`src/seo_blog_assistant/schemas.py` validate the request, research evidence,
plan, images, and final article.

The workflow:

1. Assesses whether model knowledge is sufficient or web research is
   appropriate, and classifies the topic.
2. Searches with Tavily when the selected category or request requires it and a
   search provider is configured. Research evidence retains source metadata.
3. Creates an SEO plan with a title, keywords, headings, and word budgets.
4. Writes either sequential Markdown batches (`batched`, the default) or
   concurrent planned sections (`per_section`).
5. Processes image markers according to the configured mode: copyable
   placeholders by default, optional Hugging Face image generation, or no
   images.
6. Validates and assembles an ordered `ArticleResult` for the API or UI.

The first research-assessment decision is a routing choice, not a claim that the
LLM is the sole source of every statement. In both research modes, the LLM
writes the prose. When web research is selected, retrieved sources are supplied
as additional context. The UI labels these modes separately.

### Reliability and user-facing behavior

- Text and search provider calls use bounded retry behavior and sanitized
  errors; text calls also share bounded concurrency controls.
- Section order is derived from the SEO plan rather than task completion order.
- Image generation is isolated from article writing; image failures do not
  discard the article.
- Unsupported figure checks add warnings rather than presenting unverified
  figures as confirmed.
- The Streamlit UI displays workflow errors, article warnings, research mode,
  and image outcomes.
- The Streamlit sidebar accepts a user's Groq API key as a password field. The
  key overrides the server-configured text key for that UI request. The UI
  builds a workflow for each submission instead of caching one shared workflow.
  The key is not written to `.env` or logged. It is still sent to Groq by the
  application server, so users should only enter it on a deployment they
  trust.

## Errors encountered and resolutions

| Observed issue | Resolution |
| --- | --- |
| Hugging Face image requests failed, including provider authorization/model availability errors. | Made image provider/model configuration explicit, added sanitized status-specific hints, and kept image generation optional. Placeholder mode is the default and does not construct an image provider. |
| The UI research badge could imply an article was written only from web pages. | Changed it to distinguish `LLM knowledge · No web search` from `LLM-written · Web research assisted`; documented that the LLM writes in both modes. |
| Article validation rejected an `ArticleResult` instance with Pydantic's `model_type` error. | The application now converts a Pydantic model from a different class identity to Python data before validating against the current `ArticleResult` schema. A regression test reproduces this duplicate-class case. |
| Groq returned transient connection errors and HTTP 429 rate limits. | The text adapter retries eligible transient failures with bounded attempts and provider retry delays. The supplied logs showed the reported requests succeeding after retries; persistent rate limits may still require waiting or lowering usage. |
| Generated Markdown sometimes fell outside the expected word-count range or returned headings out of order. | The batched writer checks headings and approximate word budgets, retries once when required headings are missing, and records warnings for non-fatal order or length deviations. Warnings indicate quality differences can remain even when article validation succeeds. |

These notes summarize observed development issues, not a guarantee that every
provider or model configuration will work. Provider credentials, quota, model
availability, and generated content can still cause request-specific failures.

## Verification snapshot

At the time this document was added:

- The complete offline test suite passed: **193 passed**.
- Python source and tests compiled successfully.
- `git diff --check` passed.
- No changes were made under `backend/`.

Run the UI from the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m streamlit run src\seo_blog_assistant\ui.py
```

Run the tests with:

```powershell
python -m pytest
```

For setup, configuration, API routes, and full workflow details, see
`README.md` and `flow.md`.
