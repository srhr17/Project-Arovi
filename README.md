# Project Arovi

**Arovi** is a multi-agent, ADK-powered public-health briefing system built for the Google Agents Intensive capstone.

It showcases:

- **Multi-agent system**
  - LLM agents (`LlmAgent`) for ingestion, classification, trend analysis, drafting, risk checking.
  - **ParallelAgent** for concurrent regional ingestion (global / US / state / city).
  - **SequentialAgent** for a fixed pipeline: ingestion → classification → trends → drafting → combining → safety loop → metrics.
  - **LoopAgent** for iterative risk-check → redraft safety refinement.
  - A **CustomAgent** (`MetricsAgent`) for observability/metrics.

- **Tools**
  - Built-in **Google Search** tool (`google_search`) for public-health news ingestion. :contentReference[oaicite:1]{index=1}  
  - A **custom FunctionTool** (`filter_and_dedupe_tool`) for deterministic news filtering and deduplication.

- **Sessions & Memory**
  - Uses `InMemorySessionService` and `Runner` to manage sessions and agent execution. :contentReference[oaicite:2]{index=2}  
  - Designed to be compatible with `VertexAiMemoryBankService` (Memory Bank) for long-term memory (see comments in `runner.py`). :contentReference[oaicite:3]{index=3}  

- **Context engineering**
  - Uses structured state keys (`items_*`, `tagged_items`, `trend_notes`, `history_summary`, etc.).
  - Includes a “history summary” hook (placeholder) to compact past runs into a small context string when integrating Memory Bank.

- **Observability: Logging & Metrics**
  - `MetricsAgent` (custom `BaseAgent` subclass) computes simple metrics
    (`items_*_count`, `tagged_items_count`, `risk_issue_count`), writes them to
    `session.state["metrics_summary"]`, and emits a human-readable `Event` for tracing.

## Project Purpose

Arovi generates a **daily public-health briefing** for a city/date:

- Collects global, national (US), state, and city-level health news.
- Classifies items by topic, sentiment, and public-health relevance.
- Detects trends, risks, and positive developments.
- Produces a structured, calm, non-political briefing with:
  - Regional sections: Global → U.S. → State → City
  - A “Good News” section
  - A short “Public Health Fun Fact”

The tone is **warm, factual, non-alarmist**. It explicitly avoids:

- Political commentary or election-related content.
- Sensational or speculative claims.
- Unverified rumors or social-media–only sources.

## Architecture (high level)

The root agent `arovi_root_agent` delegates to `arovi_workflow_agent` (a `SequentialAgent`):

1. **Parallel ingestion** (`ingestion_parallel_agent`)
   - `global_ingestion_agent`
   - `us_ingestion_agent`
   - `state_ingestion_agent`
   - `city_ingestion_agent`

2. **Classification** (`classification_agent`)
3. **Trend analysis** (`trend_agent`)
4. **Drafting** (`drafting_agent`)
5. **Combining sections** (`combiner_agent`)
6. **Safety refinement loop** (`risk_loop_agent`)
   - `risk_checker_agent`
   - `redraft_agent`
7. **Metrics** (`MetricsAgent`)

Each stage writes structured outputs into `session.state` via `output_key`, enabling **pause/resume** and long-running flows via ADK’s sessions & runner.

## Running Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your GenAI / Vertex environment (Google AI Studio or Vertex AI):

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
# and optionally:
# export GOOGLE_GENAI_USE_VERTEXAI=TRUE
# export GOOGLE_CLOUD_PROJECT="your-project"
# export GOOGLE_CLOUD_LOCATION="us-central1"
```

3. Run the test harness:

```bash
python -m arovi_agent.runner
```


You should see Arovi generate a sample public-health briefing for Chicago (or whatever city you set in runner.py).