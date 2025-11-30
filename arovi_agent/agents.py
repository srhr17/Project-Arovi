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
from google.genai import types

from .models import NewsItemList, TrendNotes
from .tools import google_search, filter_and_dedupe_tool


# Use a Gemini 2.x model for google_search per ADK docs.
MODEL_NAME = "gemini-2.0-flash"
# Keep this in sync with runner.py APP_NAME
APP_NAME = "project_arovi_app"


# ---------------------------------------------------------------------------
# Helpers used by parser agents
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    """
    Best-effort extraction of a JSON object from a model response.
    Handles cases like ```json ... ``` or extra prose by:
    - finding the first '{'
    - finding the last '}'
    - returning that slice.
    """
    if not text:
        return "{}"
    # Strip code fences if present
    if "```" in text:
        # keep only between first and last fence
        parts = text.split("```")
        # heuristic: the middle part is more likely the JSON
        # but we still extract by braces
        text = "".join(parts[1:-1]) if len(parts) >= 3 else parts[-1]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return text[start : end + 1]


# ---------------------------------------------------------------------------
# 1. Ingestion agents (LLM + Google Search, raw JSON strings)
# ---------------------------------------------------------------------------

def _base_ingestion_instruction(region_label: str) -> str:
    return f"""
You are a public-health news ingestion agent for the {region_label} region.

Your ONLY job is to turn google_search RESULTS into structured items.
You MUST NOT invent or infer any health events.

Data source:
- You will call the google_search tool.
- You MUST treat its returned results as the ONLY ground truth.
- You may NOT use any prior knowledge about public health events.

For EACH search result you keep:
- region:
    - "global" for worldwide items
    - "national" for United States–wide items
    - "state" for items clearly about the given state
    - "city" for items clearly about the given city
- title:
    - EXACTLY the article title from search_results (verbatim).
- source:
    - Prefer the publisher domain or the text before the "–" in the title/snippet.
- url:
    - EXACTLY the URL from the search result.
- published_date:
    - ONLY if clearly shown in the search result (e.g., "Nov 26, 2025").
    - If not shown, use an empty string "".
- summary:
    - EXACTLY the snippet from the search result, or the first sentence of it.
    - Do NOT rewrite or paraphrase.
- topic:
    - One of:
      infectious_disease, environment, mental_health,
      health_systems, injury_prevention, other
    - Choose based ONLY on the title/snippet.
- sentiment:
    - One of: positive, neutral, negative
    - Based ONLY on the title/snippet.
- public_health_relevance:
    - 1 short sentence explaining why this matters.
    - You must NOT introduce any new numbers, locations, or outcomes.
    - You may only restate what is obvious from the title/snippet, in your own words.

Filtering:
- Discard any search result that clearly has nothing to do with public health.

OUTPUT FORMAT (CRITICAL):
- Output a SINGLE JSON object:
  {{
    "items": [ <NewsItem>, <NewsItem>, ... ]
  }}
- You may wrap this JSON in ```json fences, but the content inside the braces
  MUST follow the rules above.
- Do NOT mention anything that is not visible in the search result.
"""



global_ingestion_agent = LlmAgent(
    name="global_ingestion_agent",
    model=MODEL_NAME,
    description="Ingests global public-health news.",
    instruction=_base_ingestion_instruction("global"),
    tools=[google_search],
    output_key="items_global_raw",  # raw string, will be parsed later
)

us_ingestion_agent = LlmAgent(
    name="us_ingestion_agent",
    model=MODEL_NAME,
    description="Ingests national U.S. public-health news.",
    instruction=_base_ingestion_instruction("national (United States)"),
    tools=[google_search],
    output_key="items_us_raw",
)

state_ingestion_agent = LlmAgent(
    name="state_ingestion_agent",
    model=MODEL_NAME,
    description="Ingests state-level public-health news.",
    instruction=_base_ingestion_instruction("state-level"),
    tools=[google_search],
    output_key="items_state_raw",
)

city_ingestion_agent = LlmAgent(
    name="city_ingestion_agent",
    model=MODEL_NAME,
    description="Ingests city/local public-health news.",
    instruction=_base_ingestion_instruction("city/local"),
    tools=[google_search],
    output_key="items_city_raw",
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
# 2. Classification LLM (raw -> tagged_items_raw) + ParserAgent (→ tagged_items)
# ---------------------------------------------------------------------------

classification_agent = LlmAgent(
    name="classification_agent",
    model=MODEL_NAME,
    description="Normalizes and classifies ingested news items.",
    instruction="""
You are a classifier for public-health news.

Session state contains four RAW JSON strings:
- items_global_raw
- items_us_raw
- items_state_raw
- items_city_raw

Each of these strings SHOULD be a JSON object of the form:
{
  "items": [ <NewsItem>, <NewsItem>, ... ]
}

Tasks:
1. Parse each of the four raw JSON strings.
   - If parsing fails or JSON is missing, treat it as having an empty items list.
2. Merge all lists into a single array named `merged_items`.
3. Call the `filter_and_dedupe_tool` ONCE with:
   - items: merged_items
   The tool will return:
   - filtered_items: cleaned list
   - filtered_count, original_count
4. For each filtered item:
   - Ensure `region` is one of:
       global, national, state, city
   - Normalize `topic` to one of:
       infectious_disease, environment, mental_health,
       health_systems, injury_prevention, other
   - Normalize `sentiment` to:
       positive, neutral, negative
   - If `public_health_relevance` is missing or empty, write 1–2 factual
     sentences based only on the title and snippet, with no speculation.
5. You MUST NOT introduce new news items that do not come from the merged list.
6. Return a JSON object:
   { "items": [ <NewsItem>, <NewsItem>, ... ] }

Additional HARD constraints:

- You MUST NOT introduce any new news items.
- You MUST NOT create titles that did not appear in the merged ingestion items.
- For each output item, its `title` must exactly match the `title` of some item in merged_items.
- If you generate a candidate item whose title does not exactly match any merged title, discards that candidate.


STRICT OUTPUT RULES:
- Output ONLY that JSON object.
- You MAY optionally wrap it in ```json fences.
- Do NOT add prose or comments outside the JSON.
""",
    tools=[filter_and_dedupe_tool],
    # No output_schema here; we parse ourselves
    output_key="tagged_items_raw",
)


class TaggedItemsParserAgent(BaseAgent):
    """
    Parses `tagged_items_raw` (string) into a structured dict `tagged_items`
    using the NewsItemList Pydantic model.
    """

    name: str = "tagged_items_parser_agent"
    description: str = "Parses raw JSON from classification into structured tagged_items."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = context.session.state or {}
        raw = state.get("tagged_items_raw") or ""
        json_str = _extract_json_block(raw)

        try:
            model = NewsItemList.model_validate_json(json_str)
            data = model.model_dump()
        except Exception:
            data = {"items": []}

        state["tagged_items"] = data

        text = (
            "TaggedItemsParserAgent parsed "
            f"{len(data.get('items', []))} items from classification output."
        )
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=text)],
            ),
        )


tagged_items_parser_agent = TaggedItemsParserAgent()


# ---------------------------------------------------------------------------
# 3. Trend analysis LLM (→ trend_notes_raw) + ParserAgent (→ trend_notes)
# ---------------------------------------------------------------------------

trend_agent = LlmAgent(
    name="trend_agent",
    model=MODEL_NAME,
    description="Identifies trends, risks, and positive developments from tagged_items.",
    instruction="""
You are a public-health trend analyst.

You will receive `tagged_items` in session state: a dict with key "items",
which is an array of NewsItem-like objects.

Rules:
- You MUST base all observations ONLY on tagged_items.items.
- You MUST NOT introduce new outbreaks, locations, campaigns, or statistics
  that are not represented in those items.
- If data is sparse, you may say that explicitly (e.g., "Data is limited today...").

Tasks:
1. Identify notable trends or clusters by topic, region, and sentiment.
2. Highlight:
   - Emerging or ongoing risks
   - Positive developments (success stories, improvements)
   - Any important caveats or gaps.
3. Use a calm, non-alarmist tone and avoid speculation.

Return a JSON object with keys:
- key_trends: list of bullet points
- risks: list of bullet points
- positive_developments: list of bullet points
- notes_for_briefing_writer: short free text to guide briefing writing.

STRICT OUTPUT RULES:
- Output ONLY that JSON object.
- You MAY wrap it in ```json fences.
""",
    output_key="trend_notes_raw",
)


class TrendNotesParserAgent(BaseAgent):
    """
    Parses `trend_notes_raw` into `trend_notes` using the TrendNotes model.
    """

    name: str = "trend_notes_parser_agent"
    description: str = "Parses raw JSON from trend_agent into structured trend_notes."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = context.session.state or {}
        raw = state.get("trend_notes_raw") or ""
        json_str = _extract_json_block(raw)

        try:
            model = TrendNotes.model_validate_json(json_str)
            data = model.model_dump()
        except Exception:
            data = {
                "key_trends": [],
                "risks": [],
                "positive_developments": [],
                "notes_for_briefing_writer": "",
            }

        state["trend_notes"] = data

        text = (
            "TrendNotesParserAgent parsed trend notes with "
            f"{len(data.get('key_trends', []))} key trends."
        )
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=text)],
            ),
        )


trend_notes_parser_agent = TrendNotesParserAgent()


# ---------------------------------------------------------------------------
# 4. Drafting agent (writes full briefing_draft)
# ---------------------------------------------------------------------------

drafting_agent = LlmAgent(
    name="drafting_agent",
    model=MODEL_NAME,
    description="Drafts the full public-health briefing in Markdown.",
    instruction="""
You are Arovi, a calm public-health daily briefing writer.

You MUST base all content ONLY on:
- tagged_items["items"]: the array of NewsItem objects.
- trend_notes: structured summary of patterns in tagged_items.items.

You MUST NOT introduce:
- New events, locations, or statistics not implied by tagged_items.items.
- Political commentary or election-related content.
- Speculative or sensational statements.

If tagged_items["items"] is sparse, write a shorter briefing and be transparent
(e.g., "Today's available public-health news for this area is limited...").

Write a single, self-contained Markdown document with this structure:

# Daily Public-Health Briefing for <City>, <State> — <Date>

## Global
- Summarize global items relevant to the city or context.

## United States
- Summarize national U.S. items (if applicable).

## <State>
- Summarize state-level items (if applicable).

## <City>
- Summarize city/local items (if applicable).

## Good News
- Highlight uplifting or positive developments drawn from tagged_items.items.

## Public Health Fun Fact
- A short, neutral, educational fact related to public health.
- This may be general (not tied to the news items), but keep it factual and
  non-controversial. Do not make up speculative science.

Tone:
- Warm, calm, factual, non-alarmist.
- Avoid technical jargon when possible; explain in plain language.

STRICT CONTENT RULES:

- Every concrete claim (e.g., “X disease cases reported”, “Y deaths”, “grant terminated”) MUST be directly traceable to one or more items in tagged_items["items"].
- You MUST NOT introduce any event, disease, outbreak, grant, funding decision, or statistic that is not present in tagged_items["items"]. If it is unclear from those items, do not mention it.
- You MUST NOT change counts, dates, or locations from those items.
- Prefer to quote or closely paraphrase the `summary` field in the items.
- If tagged_items["items"] is empty or sparse for some region, explicitly say there is limited or no news for that region today instead of inventing examples.


Output ONLY the final Markdown text, with no extra commentary, and it will be
stored in session state as `briefing_draft`.
""",
    output_key="briefing_draft",
)


# ---------------------------------------------------------------------------
# 5. Risk-check LLM (→ risk_report_raw) + ParserAgent (→ risk_report)
# ---------------------------------------------------------------------------

risk_checker_agent = LlmAgent(
    name="risk_checker_agent",
    model=MODEL_NAME,
    description="Checks briefing for political, speculative, or sensational content.",
    instruction="""
You are a safety reviewer for Arovi's public-health briefing.

Session state contains:
- briefing_draft (original briefing)
- briefing_revised (may exist from a previous iteration)

Steps:
1. Take the current briefing text:
   - If `briefing_revised` exists, use that.
   - Otherwise use `briefing_draft`.
2. Scan for content that:
   - Includes political or election-related commentary.
   - Advocates for specific political parties or policies.
   - Uses speculative, fear-inducing, or sensational language.
   - Makes unverifiable or unsourced health claims.
3. For each issue, propose a concrete fix (rephrase, soften, or remove).

Return a JSON object and nothing else:
{
  "is_safe": true/false,
  "issues": [
    {
      "type": "political" | "speculative" | "sensational" | "unsupported_claim",
      "excerpt": "<short excerpt>",
      "suggested_fix": "<rewrite or removal suggestion>"
    },
    ...
  ],
  "high_level_feedback": "<short narrative summary>"
}

STRICT OUTPUT RULES:
- Output ONLY that JSON object.
- You MAY wrap it in ```json fences.
""",
    output_key="risk_report_raw",
)


class RiskReportParserAgent(BaseAgent):
    """
    Parses `risk_report_raw` into a plain dict `risk_report`.
    """

    name: str = "risk_report_parser_agent"
    description: str = "Parses raw JSON from risk_checker_agent into risk_report dict."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = context.session.state or {}
        raw = state.get("risk_report_raw") or ""
        json_str = _extract_json_block(raw)

        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        state["risk_report"] = data

        issues = data.get("issues") if isinstance(data, dict) else []
        text = (
            "RiskReportParserAgent parsed "
            f"{len(issues) if isinstance(issues, list) else 0} issues."
        )
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=text)],
            ),
        )


risk_report_parser_agent = RiskReportParserAgent()


# Redraft agent (uses parsed risk_report)
redraft_agent = LlmAgent(
    name="redraft_agent",
    model=MODEL_NAME,
    description="Redrafts briefing to address issues found by risk_checker_agent.",
    instruction="""
You are an editor applying safety fixes to Arovi's briefing.

Session state contains:
- briefing_draft (original)
- briefing_revised (possibly revised from a previous iteration)
- risk_report (JSON with issues and suggested fixes)

Steps:
1. Take the current working text:
   - If `briefing_revised` exists, start from that.
   - Otherwise start from `briefing_draft`.
2. Apply the suggestions in `risk_report.issues`:
   - Remove or rewrite political, speculative, sensational, or unsupported content.
   - Maintain structure and overall length as much as possible.
   - Preserve the warm, calm, factual tone.
3. Ensure the final version contains no explicit political advocacy or
   election-related content.

Return ONLY the full revised briefing Markdown, and it will be stored in
session state as `briefing_revised`.
""",
    output_key="briefing_revised",
)


# LoopAgent still demonstrates the "Loop" concept (does 1 pass + we parse)
risk_loop_agent = LoopAgent(
    name="risk_loop_agent",
    description="Iteratively checks and redrafts the briefing for safety and tone.",
    sub_agents=[risk_checker_agent, risk_report_parser_agent, redraft_agent],
    max_iterations=2,
)


# ---------------------------------------------------------------------------
# 6. MetricsAgent (observability over tagged_items + risk_report)
# ---------------------------------------------------------------------------

class MetricsAgent(BaseAgent):
    """
    Custom agent for simple metrics & observability.

    It reads tagging + risk state and writes a small summary
    to session.state["metrics_summary"], and emits a human-readable Event.
    """

    name: str = "metrics_agent"
    description: str = "Computes and logs Arovi metrics."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state: Dict[str, Any] = context.session.state or {}

        tagged_items_obj = state.get("tagged_items") or {}
        items = tagged_items_obj.get("items") if isinstance(tagged_items_obj, dict) else []
        items = items or []

        risk_report = state.get("risk_report") or {}
        issues = risk_report.get("issues") if isinstance(risk_report, dict) else []
        issues = issues or []

        region_counts: Dict[str, int] = {}
        for item in items:
            if isinstance(item, dict):
                region = (item.get("region") or "unknown").lower()
            else:
                region = "unknown"
            region_counts[region] = region_counts.get(region, 0) + 1

        metrics = {
            "tagged_items_count": len(items),
            "items_by_region": region_counts,
            "risk_issue_count": len(issues) if isinstance(issues, list) else 0,
        }

        state["metrics_summary"] = metrics

        text = (
            "Arovi metrics summary:\n```json\n"
            + json.dumps(metrics, indent=2)
            + "\n```"
        )

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=text)],
            ),
        )


metrics_agent = MetricsAgent()


# ---------------------------------------------------------------------------
# 7. Workflow (SequentialAgent) and Root Agent
# ---------------------------------------------------------------------------

arovi_workflow_agent = SequentialAgent(
    name="arovi_workflow_agent",
    description=(
        "End-to-end Arovi pipeline: ingestion -> classification -> "
        "parsing -> trends -> parsing -> drafting -> risk loop -> metrics."
    ),
    sub_agents=[
        ingestion_parallel_agent,
        classification_agent,
        tagged_items_parser_agent,
        trend_agent,
        trend_notes_parser_agent,
        drafting_agent,
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
2. Call the Arovi workflow tool to execute the full multi-agent pipeline.
3. Present the final briefing back to the user from session state:
   - Prefer `briefing_revised` if present, otherwise `briefing_draft`.
4. Answer follow-up questions in the same warm, factual, non-alarmist tone.

Never:
- Provide political commentary or election-related opinions.
- Speculate or exaggerate risk.
- Invent public-health events; always rely on the ingested news items.
""",
    tools=[
        AgentTool(agent=arovi_workflow_agent, skip_summarization=True)
    ],
)
