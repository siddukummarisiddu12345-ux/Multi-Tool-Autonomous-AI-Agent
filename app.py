"""Multi-Tool Autonomous AI Agent - Streamlit UI. Run: streamlit run app.py"""
from __future__ import annotations

import json

import streamlit as st

from src.agent import Agent, GeminiClient, LLMError
from src.config import Settings
from src.local_mode import run_local
from src.registry import build_registry
from src.schemas import AgentResult

AGENT_LABEL, LOCAL_LABEL = "AI agent (Gemini)", "Local tools (no API key)"
AGENT_EXAMPLES = [
    "Convert 72 F to Celsius, then tell me how many days are between 2025-01-01 and 2025-12-25.",
    "Validate this JSON and pretty-print it: {\"name\": \"Asha\", \"skills\": [\"python\", \"genai\"]}",
    "Analyze this text and list its top keywords: \"Agents choose tools. Tools return results. Agents explain results.\"",
    "What is (1250 * 1.18) - 340 / 4, and how many kilometers is that many miles?",
]
LOCAL_EXAMPLES = [
    "Calculate (12.5 + 3) * 4",
    "Convert 5 km to miles",
    "Days between 2025-01-01 and 2025-03-15",
    "What time is it in Asia/Kolkata?",
]
CSS = """
<style>
.block-container{max-width:920px;padding-top:2rem}
.mt-title{font-size:2rem;font-weight:700;margin:0}
.mt-sub{color:#6b7280;margin:.2rem 0 1rem}
.mt-pill{display:inline-block;padding:.05rem .55rem;border-radius:999px;font-size:.75rem;font-weight:600;border:1px solid}
.mt-success,.mt-ready{color:#059669;border-color:#059669}
.mt-error,.mt-setup,.mt-not_configured{color:#dc2626;border-color:#dc2626}
.mt-skipped,.mt-max_steps,.mt-no_match{color:#d97706;border-color:#d97706}
@media (max-width:640px){.block-container{padding:1rem .6rem}.mt-title{font-size:1.5rem}}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_client(settings: Settings):
    return GeminiClient(settings)


def pill(state: str, label: str) -> str:
    return f'<span class="mt-pill mt-{state}">{label}</span>'


def result_text(r: dict) -> str:
    lines = [f"Task: {r['task']}", f"Mode: {r['mode']}", f"Status: {r['status']}", "", r["answer"], "", "Steps:"]
    for s in r["steps"]:
        lines.append(f"{s['index']}. [{s['kind']}] {s['name']} - {s['status']} ({s['duration_ms']} ms): {s['summary']}")
    return "\n".join(lines)


def render_result(r: dict, key: str) -> None:
    st.markdown(f"{pill(r['status'], r['status'].replace('_', ' '))} &nbsp; "
                f"<span style='color:#6b7280;font-size:.8rem'>{r['mode']} mode · {r['duration_ms']} ms</span>",
                unsafe_allow_html=True)
    if r["status"] in ("error", "not_configured"):
        st.error(r["answer"])
    else:
        st.markdown(r["answer"])
    if r["steps"]:
        with st.expander(f"Execution log ({len(r['steps'])} steps)"):
            for s in r["steps"]:
                st.markdown(f"**{s['index']}. {s['name']}** ({s['kind']}) {pill(s['status'], s['status'])} "
                            f"`{s['duration_ms']} ms`", unsafe_allow_html=True)
                if s["input"]:
                    st.code(json.dumps(s["input"], ensure_ascii=False, indent=2), language="json")
                st.text(s["summary"])
    c1, c2 = st.columns(2)
    c1.download_button("Download JSON", json.dumps(r, indent=2, ensure_ascii=False), f"result_{key}.json",
                       "application/json", key=f"dj_{key}")
    c2.download_button("Download text", result_text(r), f"result_{key}.txt", "text/plain", key=f"dt_{key}")


def main() -> None:
    st.set_page_config(page_title="Multi-Tool Autonomous AI Agent", page_icon="🛠️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("history", [])
    settings = Settings.from_env()
    registry = build_registry(settings)

    with st.sidebar:
        st.header("Configuration")
        options = [AGENT_LABEL, LOCAL_LABEL]
        mode_label = st.radio("Mode", options, index=0 if settings.has_llm else 1,
                              help="Agent mode lets Gemini choose tools. Local mode is a fixed keyword router.")
        mode = "agent" if mode_label == AGENT_LABEL else "local"
        max_steps = st.slider("Max tool calls per task", 1, 12, settings.max_steps)
        st.markdown("**API status**")
        st.markdown(f"Gemini key {pill('ready', 'set') if settings.has_llm else pill('setup', 'missing')}<br>"
                    f"Model: `{settings.model}`<br>"
                    f"Tavily key {pill('ready', 'set') if settings.has_search else pill('setup', 'missing')}",
                    unsafe_allow_html=True)
        st.markdown("**Tools**")
        for spec in registry.specs():
            state, label = spec.status()
            st.markdown(f"`{spec.name}` {pill(state, label)}", unsafe_allow_html=True)
        if st.session_state["history"]:
            st.download_button("Download all history (JSON)",
                               json.dumps([h["result"] for h in st.session_state["history"]], indent=2, ensure_ascii=False),
                               "task_history.json", "application/json", key="dl_all")
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

    st.markdown('<p class="mt-title">Multi-Tool Autonomous AI Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="mt-sub">Describe a task in plain language. In AI mode, Gemini decides which tools to call '
                '(calculator, text analyzer, JSON validator, date/time, unit converter, web research) and explains the result.</p>',
                unsafe_allow_html=True)

    agent_blocked = mode == "agent" and not settings.has_llm
    if agent_blocked:
        st.error("GOOGLE_API_KEY is not set, so AI agent mode is unavailable. Add it to .env (see .env.example) "
                 "and restart, or switch to Local tools in the sidebar.")
    elif mode == "local":
        st.info("Local mode is a transparent rule-based router (keywords and patterns), **not** an AI model. "
                "It works without an API key.")
    if not settings.has_search:
        st.caption("Web research is not configured (TAVILY_API_KEY missing); other tools are unaffected.")

    typed = st.chat_input("Describe your task…", disabled=agent_blocked)
    prompt = typed or st.session_state.pop("pending_prompt", None)

    if not st.session_state["history"] and not prompt:
        st.markdown("#### Try an example")
        for i, ex in enumerate(AGENT_EXAMPLES if mode == "agent" else LOCAL_EXAMPLES):
            if st.button(ex, key=f"ex_{mode}_{i}", disabled=agent_blocked):
                st.session_state["pending_prompt"] = ex
                st.rerun()

    for n, item in enumerate(st.session_state["history"]):
        with st.chat_message("user"):
            st.markdown(item["result"]["task"])
        with st.chat_message("assistant"):
            render_result(item["result"], str(n))

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Working…"):
                if mode == "local":
                    result = run_local(prompt, registry, settings.secrets)
                else:
                    try:
                        client = get_client(settings)
                    except LLMError as exc:
                        result = AgentResult(task=prompt, mode="agent", status="not_configured",
                                             answer=str(exc), error=str(exc))
                    else:
                        past = [m for h in st.session_state["history"] if h["result"]["mode"] == "agent"
                                for m in ({"role": "user", "content": h["result"]["task"]},
                                          {"role": "assistant", "content": h["result"]["answer"]})]
                        result = Agent(registry, client, settings, max_steps).run(prompt, past)
            data = result.model_dump()
            render_result(data, str(len(st.session_state["history"])))
        st.session_state["history"].append({"result": data})


main()
