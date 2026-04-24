"""
api/query_planner.py

LLM call 1: convert a natural language question into a structured QueryPlan.

Public API:
    plan_query(question, history=None) -> QueryPlan
"""
import json

from openai import OpenAI
from pydantic import ValidationError

from api.schemas import QueryPlan
from config import settings

_client = OpenAI(base_url=settings.llm_base_url, api_key="not-required", timeout=300.0)

# Full attribute schema injected verbatim into the planner prompt.
_SCHEMA = """
-- scalar fields
price_range: 1 (cheap) | 2 | 3 | 4 (expensive) | null
noise_level: "quiet" | "average" | "loud" | "very_loud" | null
alcohol: "full_bar" | "beer_and_wine" | "none" | null
attire: "casual" | "dressy" | "formal" | null
wifi: "free" | "paid" | "no" | null
smoking: "no" | "outdoor" | "yes" | null
stars: float 1.0–5.0

-- boolean fields (true | false | null means "don't filter on this")
good_for_groups, takes_reservations, outdoor_seating, good_for_kids,
good_for_dancing, happy_hour, has_tv, caters, wheelchair_accessible,
dogs_allowed, byob, corkage

-- JSON sub-keys (set to true to require, omit to ignore)
ambience: romantic | intimate | classy | hipster | divey | touristy | trendy | casual | upscale
good_for_meal: breakfast | brunch | lunch | dinner | latenight | dessert
music: live | dj | jukebox | karaoke | background_music
parking: garage | street | lot | valet
"""

_SYSTEM_PROMPT = f"""You are a query planner for a New Orleans restaurant search engine.
Convert the user's question into a JSON execution plan with exactly these keys:

{{
  "intent": "find_businesses" | "question_about_business" | "compare_businesses",
  "sql_filters": {{ ... }},
  "semantic_query": "<concise search phrase for vector similarity>"
}}

sql_filters rules:
- Only include fields the user's words DIRECTLY AND EXPLICITLY state. Omit everything else.
- DO NOT infer fields from occasion words. "bachelor party", "date night", "birthday dinner",
  "bachelorette" do NOT imply attire, ambience, price_range, or any other field.
- DO NOT infer attire from any occasion or vibe description.
- DO NOT infer noise_level unless the user uses words like "quiet", "loud", "lively".
- When in doubt, omit the field. Fewer filters is better than wrong filters.
- Scalar fields: set to a single value or a list of values (for OR logic).
- Range: {{"lte": N}} or {{"gte": N}} for price_range or stars.
- Boolean fields: true to require, false to exclude, omit to ignore.
- JSON sub-key fields (ambience, good_for_meal, music, parking): use {{"key": true}} for required sub-keys.
- Do not invent filter values — use only the values listed in the schema below.

semantic_query rules:
- Do NOT repeat terms already captured by sql_filters. SQL handles structured facts; semantic handles experiential, qualitative language that only lives in review text.
- noise_level is in sql_filters → do not put "loud" or "quiet" in semantic_query.
- good_for_groups is in sql_filters → do not put "large group" in semantic_query.
- good_for_meal is in sql_filters → do not put meal names in semantic_query.
- Focus semantic_query on: atmosphere, vibe, food quality, experience narratives, occasion fit.

Available fields:
{_SCHEMA}

Examples:

User: "bachelor party spot, loud, handles large groups"
Answer: {{"intent": "find_businesses", "sql_filters": {{"noise_level": ["loud", "very_loud"], "good_for_groups": true}}, "semantic_query": "fun lively bachelor party celebration great time"}}

User: "cheap brunch place with outdoor seating"
Answer: {{"intent": "find_businesses", "sql_filters": {{"good_for_meal": {{"brunch": true}}, "outdoor_seating": true, "price_range": {{"lte": 2}}}}, "semantic_query": "relaxed weekend morning patio mimosas"}}

Return ONLY the JSON object. No explanation, no markdown, no code block."""

_RETRY_REMINDER = """Your previous response was not valid JSON matching the required schema.
Return ONLY the JSON object with keys: intent, sql_filters, semantic_query.
No explanation, no markdown fences."""


def _call_llm(messages: list[dict]) -> str:
    response = _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def _parse(raw: str) -> QueryPlan:
    # Strip markdown code fences if the model wraps the JSON despite instructions.
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return QueryPlan.model_validate(json.loads(raw.strip()))


def plan_query(question: str, history: list[dict] | None = None) -> QueryPlan:
    """
    Convert a natural language question into a QueryPlan.

    Args:
        question: The user's current question.
        history:  Last ≤3 conversation turns as [{"role": ..., "content": ...}, ...].
                  Used by v2 to resolve references like "the first one" or "anything cheaper?".

    Returns:
        QueryPlan with intent, sql_filters, and semantic_query.

    Raises:
        ValueError: If the LLM returns malformed output after one retry.
    """
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])  # at most 3 turns = 6 messages (user+assistant each)
    messages.append({"role": "user", "content": question})

    raw = _call_llm(messages)
    try:
        return _parse(raw)
    except (json.JSONDecodeError, ValidationError, KeyError):
        pass

    # Retry once with an explicit schema reminder.
    messages.append({"role": "assistant", "content": raw})
    messages.append({"role": "user", "content": _RETRY_REMINDER})
    raw = _call_llm(messages)
    try:
        return _parse(raw)
    except (json.JSONDecodeError, ValidationError, KeyError) as exc:
        raise ValueError(f"Query planner returned malformed output after retry: {raw!r}") from exc
