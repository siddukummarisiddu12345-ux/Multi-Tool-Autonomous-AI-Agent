# Multi-Tool Autonomous AI Agent

A Streamlit app in which a Gemini model, through LangChain tool calling, decides which of six tools to use, runs them (several in sequence when needed), and explains the result. Every step is logged in the UI.

## Problem statement
Plain chatbots guess at arithmetic, dates, unit conversions and live facts. This project gives an LLM a small set of validated tools and lets it choose and chain them, while keeping execution bounded, transparent and safe.

## Key features
- **Autonomous tool selection**: the model picks tools via LangChain tool calling; the app runs a bounded loop (default 6 tool calls) and blocks repeated identical calls.
- **Six tools**: calculator, text analyzer, JSON validator, date/time, unit converter, web research.
- **Local mode (no API key)**: a deterministic rule-based router. It is clearly labelled in the UI and is not an AI model or autonomous agent.
- Pydantic-validated tool inputs, structured results, execution log, session task history, JSON/text downloads, API status sidebar.
- Graceful handling of missing keys, invalid input, rate limits, timeouts, malformed model replies and unavailable tools.

## Architecture
```
app.py (Streamlit UI)
  |- src/agent.py       Gemini/LangChain tool-calling loop -> ToolRegistry
  |- src/local_mode.py  rule-based router (no LLM) -> ToolRegistry
  |- src/registry.py    discovery, Pydantic validation, contained execution
  |- src/schemas.py     tool input schemas + AgentResult/ToolStep outputs
  `- src/tools/*.py     calculator, text, JSON, date/time, conversion, web research
```

## Technologies
Python 3.12, Streamlit, LangChain, Gemini, Pydantic v2, python-dotenv, pytest, Tavily (optional).

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```
Set `GOOGLE_API_KEY` in `.env` for AI mode. `GEMINI_API_KEY` is also accepted. `TAVILY_API_KEY` is optional and enables web research. Never commit a real API key.

## Tools and example prompts
| Tool | What it does | Example |
|---|---|---|
| calculator | Safe arithmetic expression evaluator | "Calculate (12.5 + 3) * 4" |
| text_analyzer | Word, character, sentence and keyword analysis | "Analyze this text: Agents choose tools." |
| json_validator | Parse and validate JSON with required keys | "Validate this JSON: {\"a\": 1}" |
| datetime_tool | Current date/time and date differences | "Days between 2025-01-01 and 2025-03-15" |
| unit_converter | Length, weight, temperature and volume | "Convert 5 km to miles" |
| web_research | Real Tavily search results with URLs | "Search for the latest Python release" |

## Run tests
```bash
python -m pytest -q
```
The automated test suite uses deterministic fake model responses and mocked HTTP; no API key is needed.

## Deploy on Render
This repository includes a `render.yaml` Blueprint and a `.python-version` pin:
1. Sign in to Render and choose **New → Blueprint**.
2. Connect this GitHub repository and select the `main` branch.
3. Render reads `render.yaml`. When prompted, provide `GOOGLE_API_KEY` from Google AI Studio as a secret. Never put it in GitHub.
4. Start the deployment and open the generated `onrender.com` URL.
5. Optionally add `TAVILY_API_KEY` in the Render service's Environment settings to enable web research.

For a manual Render Web Service, use build command `pip install -r requirements.txt` and start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`. The configured Python version is 3.12.8.

## Security and limitations
Tool inputs are validated and execution is bounded; the project does not use `eval` or `exec` for user calculations. API keys are loaded from environment variables and should be stored as deployment secrets. Chat text and tool results are sent to Google when AI mode is used; search queries are sent to Tavily when web research is used. History is session-only and there is no user authentication or persistent storage.
