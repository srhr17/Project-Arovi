import asyncio
import os
from typing import Any, Dict, Tuple

import streamlit as st
from google.adk.runners import InMemoryRunner

from arovi_agent.agents import arovi_root_agent, APP_NAME

# Try to load .env for GOOGLE_API_KEY
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is not set.\n\n"
        "Set it in your environment (export GOOGLE_API_KEY=...) "
        "or create a .env file in the project root with:\n\n"
        "GOOGLE_API_KEY=your_real_key_here"
    )
    st.stop()


# -------------------------------------------------------------------
# Core: run Arovi once and return (briefing_markdown, state_dict)
# -------------------------------------------------------------------
async def _run_arovi_once_async(
    request: str,
    user_id: str = "streamlit-user",
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the Arovi pipeline for a single natural-language request.

    1. Create session
    2. run_debug(...)
    3. Read session.state
    4. Return final_briefing (if present) + full state dict for debug
    """

    runner = InMemoryRunner(
        app_name=APP_NAME,
        agent=arovi_root_agent,
    )

    # 1) Create new session
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
    )

    # 2) Run the root agent (ignore return value, we read state)
    try:
        await runner.run_debug(
            request,
            session=session,
            user_id=user_id,
            verbose=True,
        )
    except TypeError:
        try:
            await runner.run_debug(
                request,
                user_id=user_id,
                verbose=True,
            )
        except TypeError:
            await runner.run_debug(request)

    # 3) Get state object
    state_obj = getattr(session, "state", None)

    # 4) Convert to dict for debug
    state_dict: Dict[str, Any] = {}
    if state_obj is not None:
        try:
            state_dict = state_obj.to_dict()
        except Exception:
            try:
                state_dict = dict(state_obj)
            except Exception:
                state_dict = {}

    # 5) Prefer final_briefing if present, else fall back to any 'briefing*' key
    briefing_md = ""

    # First try explicit final_briefing (what MetricsAgent now writes)
    for k, v in state_dict.items():
        if "final_briefing" in str(k):
            if isinstance(v, str) and v.strip():
                briefing_md = v
                break

    # If that failed, fall back to any 'briefing*' key just in case
    if not briefing_md:
        for k, v in state_dict.items():
            if "briefing" in str(k).lower():
                if isinstance(v, str) and v.strip():
                    briefing_md = v
                    break

    return briefing_md, state_dict




def run_arovi_once(request: str, user_id: str = "streamlit-user") -> Tuple[str, Dict[str, Any]]:
    """Sync wrapper for Streamlit."""
    return asyncio.run(_run_arovi_once_async(request=request, user_id=user_id))


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Project Arovi — Public Health Briefing",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Project Arovi")
st.caption("Multi-agent public-health daily briefing (Google ADK)")

st.markdown(
    """
Arovi collects **real public-health news** via Google Search (through ADK tools),
classifies and analyzes it with a multi-agent workflow, and generates a calm,
non-alarmist daily briefing.
"""
)

with st.sidebar:
    st.header("Briefing Parameters")

    city = st.text_input("City", value="Chicago")
    state_region = st.text_input("State / Region", value="Illinois")
    country = st.text_input("Country", value="United States")

    date_str = st.text_input("Date", value="today")

    st.markdown("---")
    user_tone = st.selectbox(
        "Tone preference",
        [
            "Calm and neutral (default)",
            "Warm and encouraging",
            "Very concise",
            "More detailed explanation",
        ],
        index=0,
    )

    run_button = st.button("🔍 Generate Daily Briefing", type="primary")

st.markdown("## 🧾 Daily Briefing")

if "last_briefing" not in st.session_state:
    st.session_state["last_briefing"] = ""
if "last_state" not in st.session_state:
    st.session_state["last_state"] = {}

if run_button:
    full_request = (
        f"Generate a calm, public-health daily briefing for {city} in the "
        f"state/region {state_region} in {country} for {date_str}. "
        "Avoid politics, focus on trustworthy public health information, "
        "and use a warm, non-alarmist tone."
    )
    if user_tone != "Calm and neutral (default)":
        full_request += f" Apply this tone preference: {user_tone}."

    st.info("Running Arovi multi-agent pipeline…")
    with st.spinner("Contacting Arovi agents and Google Search tools…"):
        try:
            briefing_md, state_dict = run_arovi_once(full_request, user_id="streamlit-user")
            st.session_state["last_briefing"] = briefing_md or ""
            st.session_state["last_state"] = state_dict or {}
            st.success("Briefing generated.")
        except Exception as e:
            st.error(f"Error while running Arovi: {e}")
            st.stop()

briefing_md = st.session_state.get("last_briefing") or ""
if briefing_md.strip():
    st.markdown("### 📄 Daily Briefing")
    st.markdown(briefing_md)
else:
    st.warning(
        "No briefing generated. "
        "Check that the Arovi pipeline is populating `briefing_draft` / `briefing_revised` in state."
    )

with st.expander("🔍 Debug: Session state / metrics (ADK)"):
    state_dict = st.session_state.get("last_state") or {}
    st.write("Raw state dictionary (namespaced keys):")
    st.json(state_dict)

    metrics_keys = [k for k in state_dict.keys() if "metrics" in k.lower()]
    if metrics_keys:
        st.markdown("#### Metrics-related entries")
        for k in metrics_keys:
            st.code(f"{k}: {state_dict[k]}")
