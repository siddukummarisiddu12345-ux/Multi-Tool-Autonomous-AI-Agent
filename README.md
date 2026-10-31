# Multi-Tool Autonomous AI Agent

A Streamlit app in which a Gemini model, through LangChain tool calling, decides which of six tools to use, runs them (several in sequence when needed), and explains the result. Every step is logged in the UI.

## Problem statement
Plain chatbots guess at arithmetic, dates, unit conversions and live facts. This project gives an LLM a small set of validated tools and lets it choose and chain them, while keeping execution bounded, transparent and safe.

## Key features
- **Autonomous tool selection**: the model picks tools via LangChain tool calling; the app runs a bounded loop (default 6 tool calls) and blocks repeated identical calls.
- **Six tools**: calculator, text analyzer, JSON validator, date/time, unit converter, web research.
- **Local mode (no API key)**: a deterministic *rule-based router* (keywords and regex). It is clearly labelled in the UI and is **not** an AI model or autonomous agent.
- Pydantic-validated tool inputs, structured results, execution log, session task history, JSON/text downloads, API status sidebar.
- Graceful handling of missing keys, invalid input, rate limits, timeouts, malformed model replies and unavailable tools.

## Architecture
```
app.py (Streamlit UI)
  |- src/agent.py       LLM loop (GeminiClient via LangChain) -> ToolRegistry
  |- src/local_mode.py  rule-based router (no LLM) -> ToolRegistry
  |- src/registry.py    discovery, Pydantic validation, contained execution
  |- src/schemas.py     tool input schemas + AgentResult/ToolStep outputs
  |- src/tools/*.py     pure-Python tool logic (raise ToolError)
  `- src/config.py, src/security.py
```
LangChain is confined to `GeminiClient`; tool logic has no framework dependency, so tools and the agent loop are unit-tested with a scripted fake LLM.

## Technologies
Python 3.11+, Streamlit, LangChain (`langchain-core`, `langchain-google-genai`), Gemini, Pydantic v2, python-dotenv, pytest, Tavily (optional, REST).

## Project structure
```
multi-tool-autonomous-ai-agent/
├── app.py  requirements.txt  README.md  .env.example  .gitignore  pytest.ini  conftest.py  render.yaml
├── .streamlit/config.toml
├── src/  __init__.py config.py agent.py local_mode.py registry.py schemas.py security.py
│   └── tools/  __init__.py base.py calculator.py text_analyzer.py json_validator.py
│               datetime_tool.py unit_converter.py web_research.py
└── tests/  test_calculator.py test_text_analyzer.py test_json_validator.py test_datetime_tool.py
            test_unit_converter.py test_web_research.py test_registry.py test_config.py
            test_local_mode.py test_agent.py
```

## Setup
```
python -m venv .venv
```
Windows: `.venv\Scripts\activate` — Linux/macOS: `source .venv/bin/activate`
```
pip install -r requirements.txt
```

## API key configuration
Copy `.env.example` to `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) | for AI mode | Free key from https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | no | Defaults to `gemini-2.5-flash`; confirm it is available to your key and change if needed |
| `TAVILY_API_KEY` | no | Enables `web_research` (https://app.tavily.com/) |
| `MAX_STEPS`, `MAX_RETRIES` | no | Tool-call limit (1–12, default 6) and retry limit (default 2) |

Without keys the app still launches: AI mode shows a clear notice, Local mode works, and web research reports it is not configured.

## Run
```
streamlit run app.py
python -m pytest -q
```

## Tools and example prompts
| Tool | What it does | Example |
|---|---|---|
| `calculator` | Safe AST evaluator (no `eval`/`exec`): `+ - * / // % ** ^`, parentheses, sqrt/abs/round/floor/ceil | "Calculate (12.5 + 3) * 4" |
| `text_analyzer` | Words, characters, sentences, paragraphs, keyword frequencies | `Analyze this text: "Agents choose tools."` |
| `json_validator` | Parse, report errors with line/column, check required keys, pretty-print | `Validate this JSON: {"a": 1}` |
| `datetime_tool` | Current date/time (optional IANA timezone) and date differences | "Days between 2025-01-01 and 2025-03-15" |
| `unit_converter` | Length, weight, temperature, volume | "Convert 5 km to miles" |
| `web_research` | Real Tavily results with URLs; never fabricated | "Search for the latest Python release" |

Multi-step AI-mode example: *"Convert 72 F to Celsius, then tell me how many days are between 2025-01-01 and 2025-12-25."*

## Testing
`python -m pytest -q` runs deterministic tests: every tool with valid and invalid input, registry/schema validation, the local router, and the agent (tool selection, multi-step chains, step limits, duplicate blocking, missing key, simulated API failures/rate limits, malformed replies) using a scripted fake LLM and mocked HTTP. No API key or network is needed.

## Deploy on Streamlit Community Cloud
1. Push the project to a GitHub repository.
2. At https://share.streamlit.io create an app, choose the repo, branch `main`, main file `app.py`.
3. Under *Advanced settings → Secrets* add: `GOOGLE_API_KEY = "..."`, optionally `GEMINI_MODEL` and `TAVILY_API_KEY`. Top-level secrets are exposed as environment variables.

## Deploy on Render (alternative)
`render.yaml` is included. In Render choose *New → Blueprint* (or *Web Service*) and point it at your GitHub repo.
Build: `pip install -r requirements.txt`. Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`.
Add `GOOGLE_API_KEY` (and optionally `TAVILY_API_KEY`) in the service's *Environment* tab; never commit them. Free instances sleep when idle.

## Security
No `eval`/`exec`, shell or file access; tools are a fixed registry with validated inputs; limits on input size, expression complexity, tool calls, timeouts and retries; secrets come from the environment and are redacted from logs, results and errors. Chat text and tool results are sent to Google (and search queries to Tavily) when those features are used.

## Limitations and future improvements
- Session-only history; no persistence or authentication.
- Local mode is keyword-based and handles simple phrasings only.
- Web research needs a Tavily key; one search provider.
- Gemini model ids and package versions change; pin versions after your first successful install.
- Possible additions: streaming responses, file upload tool, semantic routing, LangGraph, richer unit coverage.
