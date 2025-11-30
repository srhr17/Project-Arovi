from typing import List, Dict, Any
from google.adk.tools import FunctionTool, google_search

# Built-in tool re-export so agents can import from arovi_agent.tools
google_search_tool = google_search


def _filter_and_dedupe_items_impl(
    items: List[Dict[str, Any]],
    min_relevance_len: int = 40,
) -> Dict[str, Any]:
    """
    Custom Function Tool for Arovi.

    This runs as a pure function (no direct state access). It:
      - Drops items with short 'public_health_relevance' fields.
      - Deduplicates based on (region, title).
      - Returns the filtered items + some basic counts.

    The calling agent is responsible for taking `filtered_items` from the
    tool result and storing them in session.state (e.g., as `filtered_items`).
    """
    seen = set()
    filtered: List[Dict[str, Any]] = []

    for item in items:
        title = (item.get("title") or "").strip()
        region = (item.get("region") or "").strip()
        relevance = (item.get("public_health_relevance") or "").strip()

        if not title or not region:
            continue
        if len(relevance) < min_relevance_len:
            continue

        key = (region.lower(), title.lower())
        if key in seen:
            continue
        seen.add(key)
        filtered.append(item)

    return {
        "filtered_items": filtered,
        "filtered_count": len(filtered),
        "original_count": len(items),
    }


# Wrap our function in an ADK FunctionTool so LlmAgents can call it.
filter_and_dedupe_tool = FunctionTool(func=_filter_and_dedupe_items_impl)
