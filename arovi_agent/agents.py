import json
from typing import Any, Dict, List, AsyncGenerator

from google.adk.agents import (
    Agent,
    LlmAgent,
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
    BaseAgent,
)
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.tools.agent_tool import AgentTool

from .models import NewsItem, TrendNotes, BriefingSections
from .tools import google_search_tool, filter_and_dedupe_tool


MODEL_NAME = "gemini-2.5-flash"
APP_NAME = "project_arovi_app"


# ---------------------------------------------------------------------------
# 1. Ingestion agents (LLM + Google Search)
# ---------------------------------------------------------------------------

def _base_ingestion_instruction(region_label: str) -> str:
    return f"""
You are a public-health news ingestion agent for the {region_label} region.

Your tasks:
1. Use Google Search (via the google_search tool) to find trustworthy, non-political,
   public-health–relevant news for the specified city and date (or a short range
   around that date, like ±1 day).
2. Focus on:
   - Communicable and non-communicable disease trends
   - Environmental health (air quality, heat, pollution, disasters)
   - Health systems and access to care
   - Vaccination and prevention campaigns
   - Community health initiatives
3. Avoid:
   - Partisan politics or election content
   - Speculation, rumors, or unverified social media
   - Sensational or fear-inducing framing

Output:
- A concise JSON array of news items. For each item, include:
  region, title, source, url, published_date, summary,
  topic, sentiment, public_health_relevance.

Tone:
- Calm, factual, non-alarmist.
- If you are unsure about a claim, either omit the item or clearly mark uncertainty.
"""


global_ingestion_agent = LlmAgent(
    name="global_ingestion_agent",
    model=MODEL_NAME,
    description="Ingests global public-health news.",
    instruction=_base_ingestion_instruction("global"),
    tools=[google_search_tool],
    output_key="items_global",
)

us_ingestion_agent = LlmAgent(
    name="us_ingestion_agent",
    model=MODEL_NAME,
    description="Ingests national U.S. public-health news.",
    instruction=_base_ingestion_instruction("United States"),
    tools=[google_search_tool],
    output_key="items_us",
)

state_ingestion_agent = LlmAgent(
    name="state_ingestion_agent",
    model=MODEL_NAME,
    description="Ingests state-level public-health news.",
    instruction=_base_ingestion_instruction("state-level"),
    tools=[google_search_tool],
    output_key="items_state",
)

city_ingestion_agent = LlmAgent(
    name="city_ingestion_agent",
    model=MODEL_NAME,
    description="Ingests city/local public-health news.",
    instruction=_base_ingestion_instruction("city/local"),
    tools=[google_search_tool],
    output_key="items_city",
)

ingestion_parallel_agent = ParallelAgent(
    name="ingestion_parallel_agent",
    description="Runs regional ingestion agents in parallel.",
    sub_agents=[
        global_ingestion_agent,
        us_ingestion_agent,
        state_ingestion_agent,
        city_ingestion_agent,
    ],
)

# ---------------------------------------------------------------------------
# 2. Classification & Tagging agent (LLM + FunctionTool)
# ---------------------------------------------------------------------------

classification_agent = LlmAgent(
    name="classification_agent",
    model=MODEL_NAME,
    description="Normalizes and classifies ingested news items.",
    instruction="""
You are a classifier for public-health news.

Session state will contain:
- items_global
- items_us
- items_state
- items_city

Steps:
1. Read all items from these keys and merge into a single list.
2. Call the `filter_and_dedupe_tool` with that list to get `filtered_items`.
3. For each filtered item:
   - Ensure it has region set to one of: global, national, state, city.
   - Assign a consistent topic label (e.g. infectious_disease, environment,
     mental_health, health_systems, injury_prevention, etc.).
   - Assign sentiment: positive, neutral, or negative.
   - Provide a concise public_health_relevance explanation if missing.
4. Remove off-topic items (no clear public-health relevance).
5. Write your final list of cleaned items into session state as `tagged_items`
   and respond with a short text summary (counts per region, top topics).

Important:
- Avoid politics and policy opinions.
- Do not fabricate events; if unsure, drop the item.
""",
    tools=[filter_and_dedupe_tool],
    output_key="tagged_items",
)

# ---------------------------------------------------------------------------
# 3. Trend analysis agent
# ---------------------------------------------------------------------------

trend_agent = LlmAgent(
    name="trend_agent",
    model=MODEL_NAME,
    description="Identifies trends, risks, and positive developments.",
    instruction="""
You are a public-health trend analyst.

You will see `tagged_items` in session state: a list of structured news items
with region, topic, sentiment, and public_health_relevance fields.

Tasks:
1. Identify notable trends or clusters by topic, region, and sentiment.
2. Highlight:
   - Emerging or ongoing risks
   - Positive developments (success stories, improvements)
   - Any important caveats or missing information.
3. Use a calm, non-alarmist tone and avoid speculation.

Write a small JSON-like object into `trend_notes` in session state with keys:
- key_trends: list of bullet point strings
- risks: list of bullet point strings
- positive_developments: list of bullet point strings
- notes_for_briefing_writer: free text to guide the briefing writer.

Then, respond with a short text summary of the key_trends.
""",
    output_key="trend_notes",
)

# ---------------------------------------------------------------------------
# 4. Drafting & Combining agents
# ---------------------------------------------------------------------------

drafting_agent = LlmAgent(
    name="drafting_agent",
    model=MODEL_NAME,
    description="Drafts structured briefing sections in Markdown.",
    instruction="""
You are Arovi, a calm public-health briefing writer.

Session state provides:
- tagged_items: list of classified NewsItem-like dicts.
- trend_notes: JSON-like dict with key_trends, risks, positive_developments.

Write clear, friendly Markdown sections:

1. Global
2. National (U.S.)
3. State (for the given state; if unknown, keep generic)
4. City (for the given city; if unknown, keep generic)
5. Good News (uplifting items from any region)
6. Public Health Fun Fact (short, neutral, educational; no controversy)

Tone:
- Warm, calm, factual, non-alarmist.
- No political commentary, no election-related content.
- Avoid speculative claims. If uncertain, say so briefly or omit.

Store your sections in session state under:
- section_global
- section_us
- section_state
- section_city
- section_good_news
- section_fun_fact

Then respond with a short summary of what you wrote.
""",
    output_key="draft_sections",  # just a label for downstream; content is in state keys
)

combiner_agent = LlmAgent(
    name="combiner_agent",
    model=MODEL_NAME,
    description="Combines draft sections into a single Markdown briefing.",
    instruction="""
You will receive section texts from session state:
- section_global
- section_us
- section_state
- section_city
- section_good_news
- section_fun_fact

Combine them into a single Markdown briefing with this structure:

# Daily Public-Health Briefing for <City>, <State> — <Date>

## Global
...

## United States
...

## <State>
...

## <City>
...

## Good News
...

## Public Health Fun Fact
...

Keep language accessible to a general audience and avoid fear-inducing phrasing.
Avoid politics or policy opinions.

Write the final Markdown into session state as `briefing_draft`, and respond
with a one-sentence description of the overall tone.
""",
    output_key="briefing_draft",
)

# ---------------------------------------------------------------------------
# 5. Risk-check loop (LoopAgent with 2 LLM agents)
# ---------------------------------------------------------------------------

risk_checker_agent = LlmAgent(
    name="risk_checker_agent",
    model=MODEL_NAME,
    description="Checks briefing for political, speculative, or sensational content.",
    instruction="""
You are a safety reviewer for Arovi's public-health briefing.

Session state contains:
- briefing_draft (or briefing_revised from a previous iteration)

Tasks:
1. Read the current briefing text (prefer `briefing_revised` if present,
   otherwise `briefing_draft`).
2. Identify any content that:
   - Includes political or election-related commentary.
   - Advocates for specific policies or parties.
   - Uses speculative, fear-inducing, or sensational language.
   - Makes unverifiable or unsourced health claims.
3. For each issue, suggest a concrete fix (rephrase, soften, or remove).

Write a JSON-like dict into `risk_report` in session state with:
- is_safe: true/false
- issues: list of {type, excerpt, suggested_fix}
- high_level_feedback: string

Respond with a short text description of the number of issues found.
""",
    output_key="risk_report",
)

redraft_agent = LlmAgent(
    name="redraft_agent",
    model=MODEL_NAME,
    description="Redrafts briefing to address issues found by risk_checker_agent.",
    instruction="""
You are an editor applying safety fixes to the briefing.

Use:
- briefing_draft or briefing_revised (the current text).
- risk_report (issues + suggested fixes).

Rewrite the full briefing:
- Apply suggested fixes or safe equivalents.
- Remove political content, speculation, and sensational phrasing.
- Preserve the structure and overall length as much as reasonable.
- Maintain a warm, calm, public-health–aligned tone.

Write the result to session state as `briefing_revised`, and respond with
a simple 'Revision applied' message.
""",
    output_key="briefing_revised",
)

risk_loop_agent = LoopAgent(
    name="risk_loop_agent",
    description="Iteratively checks and redrafts briefing for safety / tone.",
    sub_agents=[risk_checker_agent, redraft_agent],
    max_iterations=3,
)

# Note: Termination condition can be improved by having risk_checker set a
# flag like risk_report['is_safe'] and then using that in a custom LoopAgent
# wrapper. For capstone simplicity, we rely on max_iterations=3.


# ---------------------------------------------------------------------------
# 6. Custom MetricsAgent (BaseAgent) for observability
# ---------------------------------------------------------------------------

class MetricsAgent(BaseAgent):
    """
    Custom agent for simple metrics & observability.

    It reads ingestion + tagging + risk state and writes a small summary
    to session.state["metrics_summary"], and emits a human-readable Event.
    """

    name: str = "metrics_agent"
    description: str = "Computes and logs Arovi metrics."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = context.session.state or {}

        items_global = state.get("items_global") or []
        items_us = state.get("items_us") or []
        items_state = state.get("items_state") or []
        items_city = state.get("items_city") or []
        tagged_items = state.get("tagged_items") or []
        risk_report = state.get("risk_report") or {}
        issues = risk_report.get("issues") if isinstance(risk_report, dict) else []
        issues = issues or []

        metrics = {
            "items_global_count": len(items_global),
            "items_us_count": len(items_us),
            "items_state_count": len(items_state),
            "items_city_count": len(items_city),
            "tagged_items_count": len(tagged_items),
            "risk_issue_count": len(issues),
        }

        await context.session.update_state({"metrics_summary": metrics})

        yield Event(
            author=self.name,
            content=(
                "Arovi metrics summary:\n```json\n"
                + json.dumps(metrics, indent=2)
                + "\n```"
            ),
        )


metrics_agent = MetricsAgent()

# ---------------------------------------------------------------------------
# 7. Workflow (SequentialAgent) and Root Agent
# ---------------------------------------------------------------------------

arovi_workflow_agent = SequentialAgent(
    name="arovi_workflow_agent",
    description=(
        "End-to-end Arovi pipeline: ingestion -> classification -> trends -> "
        "drafting -> combining -> safety loop -> metrics."
    ),
    sub_agents=[
        ingestion_parallel_agent,
        classification_agent,
        trend_agent,
        drafting_agent,
        combiner_agent,
        risk_loop_agent,
        metrics_agent,
    ],
)

arovi_root_agent = Agent(
    name="arovi_root_agent",
    model=MODEL_NAME,
    description="User-facing Arovi agent that generates daily public-health briefings.",
    instruction="""
You are Arovi, a calm public-health daily briefing assistant.

When the user asks for a briefing:
1. Extract:
   - city
   - state or region (if given)
   - country (default to United States if unspecified)
   - target date (default to today if unspecified)
2. Call the `run_arovi_pipeline` tool to execute the Arovi workflow agent.
3. Present the final briefing back to the user from session state:
   - Prefer `briefing_revised` if present, otherwise `briefing_draft`.
4. Answer follow-up questions in the same warm, factual, non-alarmist tone.

Never:
- Provide political commentary or election-related opinions.
- Speculate or exaggerate risk.
""",
    tools=[
        AgentTool(
            agent=arovi_workflow_agent,
            name="run_arovi_pipeline",
            description="Runs Arovi's full multi-agent pipeline.",
            skip_summarization=True,
        )
    ],
)
