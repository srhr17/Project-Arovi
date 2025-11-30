from typing import List
from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    """Structured representation of a public-health news item."""

    region: str = Field(
        description="One of: global, national, state, city"
    )
    title: str
    source: str
    url: str
    published_date: str
    summary: str
    topic: str = Field(
        description="E.g., infectious_disease, environment, mental_health, health_systems, etc."
    )
    sentiment: str = Field(
        description="One of: positive, neutral, negative"
    )
    public_health_relevance: str = Field(
        description="Short justification of why this matters for public health."
    )


class NewsItemList(BaseModel):
    """
    Container for lists of NewsItem, used because ADK's output_schema
    must be a single BaseModel subclass (not List[NewsItem]).
    """

    items: List[NewsItem] = Field(
        default_factory=list,
        description="List of NewsItem objects.",
    )


class TrendNotes(BaseModel):
    """Summary of trends, risks, positives for drafting."""

    key_trends: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    positive_developments: List[str] = Field(default_factory=list)
    notes_for_briefing_writer: str = ""


class BriefingSections(BaseModel):
    """Logical sections of the daily briefing (kept for future use)."""

    section_global: str
    section_us: str
    section_state: str
    section_city: str
    section_good_news: str
    section_fun_fact: str
    combined_markdown: str
