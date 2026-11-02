# SEO Blog Writing Assistant

Python 3.12 application with a shared LangGraph workflow, a FastAPI endpoint,
and a Streamlit topic-to-article interface. The API and UI return the same
validated article result.

The configured text model and optional image model are open-source/open-weight
models accessed through hosted inference (Groq and Hugging Face), rather than
models run locally by this application. Image generation is opt-in; the
default placeholder mode makes no image-model call. Use provider free-tier
quotas where available; availability and limits can change. Optional web
research uses Tavily, a hosted search service rather than a model.

The default `batched` writing mode is designed to reduce text-model token use:
a typical 1,200-word article makes three text calls (research decision, SEO
plan, and one plain-Markdown writing call). Longer articles use sequential
writing passes. Set `SEO_BLOG_WRITING_MODE=per_section` to keep the previous
structured, concurrent section workflow for stronger models or higher-quota
plans. Validation retries and truncation continuations can add calls.

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
- `SEO_BLOG_TEXT_MODEL_NAME` — start with `openai/gpt-oss-20b` after confirming
  it is available in your Groq console; it supports strict JSON Schema output
- `SEO_BLOG_GROQ_API_KEY` — Groq API key. For compatibility,
  `SEO_BLOG_TEXT_MODEL_API_KEY` is also accepted.
- `SEO_BLOG_TEXT_MODEL_TIMEOUT_SECONDS` — optional request timeout (default:
  120 seconds)
- `SEO_BLOG_TEXT_MODEL_MAX_CONCURRENCY` — simultaneous text calls per process
  (default: 2; lower this to reduce bursts)
- `SEO_BLOG_TEXT_MODEL_MAX_ATTEMPTS` — retry attempts for transient failures
  (default: 4)
- `SEO_BLOG_TEXT_MODEL_MAX_COMPLETION_TOKENS` — completion cap per model call
  (default: 4096; higher values allow longer sections but can consume more of
  the per-minute token quota)
- `SEO_BLOG_TEXT_MODEL_REASONING_EFFORT` — `low`, `medium`, `high`, or unset
  (default: `low`); applied only to model IDs starting with `openai/gpt-oss`
- `SEO_BLOG_WRITING_MODE` — `batched` (default; plain Markdown, sequential
  writing calls) or `per_section` (existing structured, concurrent
  section-writing workflow; useful for stronger models or paid tiers)
- `SEO_BLOG_MAX_WORDS_PER_CALL` — maximum planned article words per batched
  writing pass (default: 1500; larger articles use sequential passes)
- `SEO_BLOG_MAX_IMAGES` — cap on image prompts or generated images per article
  (default: 3; range 0–4)
- `SEO_BLOG_IMAGE_MODE` — `placeholder` (default; no image-provider call),
  `generate` (Hugging Face inference), or `off`
- `SEO_BLOG_SEARCH_QUERIES_PER_ARTICLE` — category-aware query cap (default: 2;
  range: 1–3)
- `SEO_BLOG_TEXT_MODEL_REQUESTS_PER_SECOND` — optional paced request rate;
  unset by default

The Groq adapter applies its own retries with bounded concurrency. Free-tier
limits vary by model and account; check the Groq console's Limits page. Strict
JSON Schema decoding is supported only by selected models, so use one of the
recommended models when available.

### Topic categories and research policy

The existing first research-assessment call also classifies the topic and
suggests a content angle; it adds no text-model call. An optional `category` in
an API request overrides automatic detection. The Streamlit UI offers the same
choice, including `Auto-detect`. Category profiles shape the outline, writing
style, evidence rules, and compact research context. The response includes
`topic_category` and `content_angle`.

| Category | Research policy | Editorial focus |
| --- | --- | --- |
| `technology` | Recommended | How it works, adoption, limitations, and trade-offs |
| `sports` | Required | Recent context, sourced stats, records, history, and analysis |
| `food` | Optional | Origins, ingredients, technique, variations, and serving |
| `politics` | Required | Neutral background, institutions, timelines, evidence, and perspectives |
| `business_finance` | Required | Market context, sourced figures, risks, and outlook |
| `health_science` | Recommended | Research evidence, benefits, risks, and practical guidance |
| `travel_culture` | Optional | Culture, attractions, practical travel information, and safety |
| `entertainment` | Recommended | Works, sourced reception, cultural impact, and what is next |
| `history_education` | Optional | Timelines, causes, interpretations, and relevance |
| `general` | Optional | A balanced outline fitted to the reader and topic |

Sports, politics, and business/finance topics always require web research; if
Tavily is not configured, generation stops with a clear configuration error.
For categories where research is recommended, the app searches when a provider
is available and otherwise returns a warning that figures are unverified.
Explicit freshness or citation requirements always require research. Required
and recommended categories also get a post-writing check that warns when
figures are absent from the compact research evidence actually sent to the
writer. It never silently asserts those figures are verified.

Tavily can run up to two category-specific queries by default; set
`SEO_BLOG_SEARCH_QUERIES_PER_ARTICLE` to 1–3 to tune that cap. Results are
merged and de-duplicated. A failed follow-up query retains first-query evidence
and adds a warning.

Web research is optional for general topics, but required when the research
decision selects current information or citations:

- `SEO_BLOG_SEARCH_PROVIDER=tavily` and `SEO_BLOG_SEARCH_API_KEY` enable web
  search and retrieved source content.
- `SEO_BLOG_SEARCH_TIMEOUT_SECONDS` and `SEO_BLOG_SEARCH_MAX_RESULTS` are
  optional and default to 30 seconds and 5 results.
- `SEO_BLOG_SEARCH_MAX_ATTEMPTS` controls transient search retries (default: 3).
- `SEO_BLOG_SEARCH_QUERIES_PER_ARTICLE` defaults to 2 category-specific
  queries when the selected profile has search hints.

Images default to copyable placeholders. This mode does not construct an image
provider and does not require Hugging Face settings or a token. Set
`SEO_BLOG_IMAGE_MODE=generate` to opt into hosted image generation with an
open-source/open-weight model through Hugging Face Inference Providers (no
local model or GPU):

- `SEO_BLOG_IMAGE_PROVIDER=huggingface`
- `SEO_BLOG_IMAGE_MODEL_NAME` — hosted image model ID (recommended:
  `black-forest-labs/FLUX.1-schnell`, Apache 2.0)
- `SEO_BLOG_IMAGE_INFERENCE_PROVIDER` — provider name or `auto` (default:
  `auto`)
- `SEO_BLOG_IMAGE_NUM_INFERENCE_STEPS` — number of steps from 1 to 50
  (default: 4; set blank/`None` in Python configuration to omit)
- `SEO_BLOG_HUGGINGFACE_TOKEN` — Hugging Face access token
- `SEO_BLOG_IMAGE_TIMEOUT_SECONDS` — optional timeout (default: 120 seconds)

The Hugging Face client uses the configured provider to route through an
available service for the configured model. Automatic routing does not make an
otherwise unsupported model available. Free accounts get only about $0.10 of
Inference Providers credit per month, shared across providers. Image generation
can consume that quickly, so keep the modest per-article image cap at 3 and leave
`generate_images` false unless images are requested.

`SEO_BLOG_IMAGE_MODE` selects placeholder, generation, or off behavior;
`generate_images` remains the per-request switch and must be `true` for either
placeholders or generated images. `SEO_BLOG_MAX_IMAGES` caps prompts/images
(default 3; set 0 to disable). If the writer returns no usable prompt, the app
adds one contextual fallback prompt. In `generate` mode, image errors are
reported in `inline_images` and the aggregate `image_result.status`; they do
not discard the article.

### Adding images by hand

1. Run the article with the default `placeholder` mode and set
   `generate_images=true` in the request.
2. Copy the prompt from each `IMAGE PLACEHOLDER` block into your image
   generator.
3. Download the image and replace the whole block with
   `![alt text](path-or-url)`.
4. Or download the Markdown without placeholders if you want to publish
   without images.

Each placeholder prompt is also returned in `image_placeholders`; the UI shows
the prompt in a copyable code block and offers downloads with and without
placeholder blocks.

### Troubleshooting images

Run a real provider check from the project root after configuring a Hugging Face
token with permission to call Inference Providers:

```powershell
python scripts/check_image_provider.py
```

The script writes `test_image.png` in the current directory and refuses to
overwrite an existing file. Failures are printed with a sanitized hint; the
token and prompt are not printed.

| HTTP status | Meaning / next step |
| --- | --- |
| 401 / 403 | Token rejected or lacks permission; check `SEO_BLOG_HUGGINGFACE_TOKEN` permissions |
| 402 | Hugging Face monthly Inference Providers credits exhausted |
| 404 / 422 | Model is not available through the selected provider; check model ID and provider |
| 429 | Provider rate limit reached |
| 5xx | Hosted inference provider error; retry later or select another available provider |

## Run the API

```powershell
python -m uvicorn --app-dir src seo_blog_assistant.api:app --reload
```

Submit `POST /articles` with JSON such as:

```json
{
  "topic": "Practical home energy savings",
  "audience": "Homeowners",
  "category": "technology",
  "target_words": 1200,
  "require_citations": true
}
```

`category` is optional and accepts the ids in the category table; omit it to use
the research-assessment model's category detection. The request and returned
article use the shared Pydantic schemas.
Invalid requests return HTTP 422. Missing provider configuration and research
provider failures return HTTP 503. Groq token/rate limits return HTTP 429;
other Groq upstream errors return HTTP 502. Research evidence is trimmed to
four 600-character snippets in text-model prompts; URLs and publishers are
omitted from prompts while full source metadata remains in the workflow.
The batched writer uses short `S1`-style citation labels internally and maps
them back to validated source IDs in the result. Safe model-output repairs and
validation warnings are included in the response's `warnings` array. The
article response contains metadata, detected `topic_category` and
`content_angle`, ordered Markdown, cited source metadata, image prompt
metadata, and image results.

In the default placeholder mode, Markdown includes copyable
`IMAGE PLACEHOLDER (img_1)` blocks, with matching `image_placeholders[]`
metadata. In `generate` mode, successful images appear as
`![alt text](image:img_1)` placeholders; match each identifier to
`inline_images[].image_id` and use its validated Base64 PNG data to render or
replace the placeholder. Failed image markers are removed from Markdown and
reported in `inline_images` and `warnings`. The aggregate `image_result` is
preserved for existing clients.
Groq completions are capped at 4,096 tokens per model call by default. This
does not increase the account's provider quota or prevent 429s when the
organization-level per-minute limit is already exhausted.

For long-running requests, `POST /jobs` accepts the same request body and
returns `202 {"job_id": "..."}`. Poll `GET /jobs/{job_id}` for a `pending`,
`running`, `completed`, or `failed` status and its result or error. Jobs are
stored in memory only, expire after one hour, and are limited to 100 stored
jobs per process. This job store is single-process; do not use it as durable
storage or behind multiple independent worker processes.

## Run the Streamlit UI

```powershell
$env:PYTHONPATH = "src"
python -m streamlit run src\seo_blog_assistant\ui.py
```

The UI identifies whether the article used `LLM knowledge · No web search` or
`LLM-written · Web research assisted`. The latter means retrieved web sources
were provided as context; the LLM still writes the prose and may also draw on
its existing knowledge. Recommended topic categories use web research when a
search provider is configured, even if the initial model assessment considered
its general knowledge sufficient.

Before submitting, enter your personal Groq API key in the sidebar. The UI
requires it for each browser session and uses it to build that request's
workflow, replacing any text-model key from the server's environment. It is not
written to `.env` or cached as a shared workflow; use **Clear API key** when
finished. The key is sent to Groq by the application server, so only enter it
when you trust the deployment and its operator.
The project Streamlit theme is pinned to light mode for consistent contrast
across text fields, dropdowns, checkboxes, and article output.

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
