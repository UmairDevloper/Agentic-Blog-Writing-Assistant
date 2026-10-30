# SEO Blog Writing Assistant

Python 3.12 application with a shared LangGraph workflow, a FastAPI endpoint,
and a Streamlit topic-to-article interface. The API and UI return the same
validated article result.

The configured text and image models are open-source/open-weight models
accessed through hosted inference (Groq and Hugging Face), rather than models
run locally by this application. Use provider free-tier quotas where available;
availability and limits can change. Optional web research uses Tavily, a
hosted search service rather than a model.

## Windows setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set the provider values in `.env` before starting an entry point. Keep `.env`
local and never commit real credentials. The example uses placeholders; replace
them with credentials from providers you are authorized to use.

## Configuration

Text generation is required and currently uses LangChain's Groq adapter to
call a hosted open-source/open-weight model:

- `SEO_BLOG_TEXT_MODEL_PROVIDER=groq`
- `SEO_BLOG_TEXT_MODEL_NAME` — a Groq model that supports structured output
- `SEO_BLOG_GROQ_API_KEY` — Groq API key. For compatibility,
  `SEO_BLOG_TEXT_MODEL_API_KEY` is also accepted.
- `SEO_BLOG_TEXT_MODEL_TIMEOUT_SECONDS` — optional request timeout (default:
  120 seconds)

Web research is optional for general topics, but required when the research
decision selects current information or citations:

- `SEO_BLOG_SEARCH_PROVIDER=tavily` and `SEO_BLOG_SEARCH_API_KEY` enable web
  search and retrieved source content.
- `SEO_BLOG_SEARCH_TIMEOUT_SECONDS` and `SEO_BLOG_SEARCH_MAX_RESULTS` are
  optional and default to 30 seconds and 5 results.

Hosted image generation is optional and uses an open-source/open-weight image
model through Hugging Face Inference API, not a local model or GPU:

- `SEO_BLOG_IMAGE_PROVIDER=huggingface`
- `SEO_BLOG_IMAGE_MODEL_NAME` — hosted image model ID, for example
  `Qwen/Qwen-Image` when available through the Hugging Face Inference API
- `SEO_BLOG_HUGGINGFACE_TOKEN` — Hugging Face access token
- `SEO_BLOG_IMAGE_TIMEOUT_SECONDS` — optional timeout (default: 120 seconds)

The request must set `generate_images` to `true` to call the image provider.
Image errors are returned as `image_result.status="failed"` while keeping the
article available.

## Run the API

```powershell
python -m uvicorn --app-dir src seo_blog_assistant.api:app --reload
```

Submit `POST /articles` with JSON such as:

```json
{
  "topic": "Practical home energy savings",
  "audience": "Homeowners",
  "target_words": 1200,
  "require_citations": true
}
```

The request and returned article use the shared Pydantic schemas.
Invalid requests return HTTP 422. Missing provider configuration and research
provider failures return HTTP 503. Groq token/rate limits return HTTP 429;
other Groq upstream errors return HTTP 502. Research evidence is trimmed to
three 1,000-character snippets in text-model prompts to limit token usage while
the workflow retains complete source metadata and evidence. The article
response contains metadata,
ordered Markdown, source metadata, and the optional image-generation result
(including validated PNG data as Base64 when image generation succeeds).
Groq completions are capped at 2,048 tokens per model call to reduce oversized
requests; this does not increase the account's provider quota or prevent
429s when the organization-level per-minute limit is already exhausted.

## Run the Streamlit UI

```powershell
$env:PYTHONPATH = "src"
python -m streamlit run src\seo_blog_assistant\ui.py
```

## Run tests

```powershell
python -m pytest
```

Tests use fake providers and do not require API credentials or network access.

## Output and current limitations

Generated articles are returned by the API or displayed by the UI. The
application does not automatically write articles to disk, so it will not
overwrite the prototype articles under `backend/` or other user-created files.
The current text provider is Groq; other text providers require an adapter.
Research requires a configured Tavily key when selected. Image generation
depends on the selected Hugging Face model being available through hosted
inference and on the token having access. Free quotas and availability are
determined by the providers and are not guaranteed by this application.
