"""
api/sql_filter.py

Execute SQL filters against the businesses table and return matching business_ids.

Public API:
    filter_businesses(sql_filters) -> list[str] | None

Returns None when the sparse fallback is exhausted, signalling the pipeline
to skip SQL filtering and run semantic retrieval across all businesses.
"""
import sqlite3
from dataclasses import dataclass

from config import settings

_JSON_FIELDS = {"ambience", "good_for_meal", "music", "parking"}
_BOOLEAN_FIELDS = {
    "good_for_groups", "takes_reservations", "outdoor_seating", "good_for_kids",
    "good_for_dancing", "happy_hour", "has_tv", "caters", "wheelchair_accessible",
    "dogs_allowed", "byob", "corkage",
}
_SCALAR_FIELDS = {"noise_level", "alcohol", "attire", "wifi", "smoking", "price_range", "stars"}
_RANGE_OPS = {"lte": "<=", "gte": ">=", "lt": "<", "gt": ">"}

# Valid sub-keys per JSON field — LLM output is checked against these before interpolation.
_JSON_SUBKEYS: dict[str, frozenset] = {
    "ambience":     frozenset({"romantic", "intimate", "classy", "hipster", "divey", "touristy", "trendy", "casual", "upscale"}),
    "good_for_meal": frozenset({"breakfast", "brunch", "lunch", "dinner", "latenight", "dessert"}),
    "music":        frozenset({"live", "dj", "jukebox", "karaoke", "background_music"}),
    "parking":      frozenset({"garage", "street", "lot", "valet"}),
}


@dataclass
class _Condition:
    sql: str
    params: list
    confidence: int  # 1 = most niche (dropped first), 2 = core constraint


def _build_conditions(sql_filters: dict) -> list[_Condition]:
    conditions: list[_Condition] = []

    for field, value in sql_filters.items():
        if field in _JSON_FIELDS:
            # {"brunch": True, "dinner": True} → one condition per True sub-key
            if isinstance(value, dict):
                valid_subkeys = _JSON_SUBKEYS.get(field, frozenset())
                for subkey, subval in value.items():
                    if subval is True and subkey in valid_subkeys:
                        conditions.append(_Condition(
                            sql=f"json_extract({field}, '$.{subkey}') = 1",
                            params=[],
                            confidence=1,
                        ))

        elif field in _BOOLEAN_FIELDS:
            if value is True:
                conditions.append(_Condition(sql=f"{field} = 1", params=[], confidence=2))
            elif value is False:
                conditions.append(_Condition(sql=f"{field} = 0", params=[], confidence=2))

        elif field in _SCALAR_FIELDS:
            if isinstance(value, dict):
                # Range filter: {"lte": 3} or {"gte": 4.0}
                for op, operand in value.items():
                    sql_op = _RANGE_OPS.get(op)
                    if sql_op:
                        conditions.append(_Condition(
                            sql=f"{field} {sql_op} ?",
                            params=[operand],
                            confidence=2,
                        ))
            elif isinstance(value, list):
                placeholders = ",".join("?" * len(value))
                conditions.append(_Condition(
                    sql=f"{field} IN ({placeholders})",
                    params=list(value),
                    confidence=2,
                ))
            elif value is not None:
                conditions.append(_Condition(sql=f"{field} = ?", params=[value], confidence=2))

        # Fields not in any whitelist are silently ignored.

    return conditions


def _execute(conditions: list[_Condition], db_path: str) -> list[str]:
    where = " AND ".join(c.sql for c in conditions)
    params = [p for c in conditions for p in c.params]
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"SELECT business_id FROM businesses WHERE {where}", params
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _drop_lowest_confidence(conditions: list[_Condition]) -> list[_Condition]:
    """Remove the first condition with the lowest confidence score."""
    min_conf = min(c.confidence for c in conditions)
    idx = next(i for i, c in enumerate(conditions) if c.confidence == min_conf)
    return conditions[:idx] + conditions[idx + 1:]


def filter_businesses(
    sql_filters: dict,
    db_path: str | None = None,
    min_stars: float = 4.0,
) -> list[str] | None:
    """
    Return business_ids matching sql_filters.

    A minimum star rating floor (default 4.0) is always applied and is never
    relaxed by the sparse fallback — it is a quality gate, not a user constraint.

    Sparse fallback:
      - If < 3 results: drop the lowest-confidence LLM constraint, retry once.
      - If still < 3: return None (caller should use semantic-only retrieval).

    Returns None also when sql_filters is empty (no structured constraints).
    """
    if not sql_filters:
        return None

    db = db_path or settings.sqlite_path
    base = [_Condition(sql="stars >= ?", params=[min_stars], confidence=3)]
    conditions = _build_conditions(sql_filters)
    if not conditions:
        return None

    results = _execute(base + conditions, db)
    if len(results) >= 3:
        return results

    # Sparse fallback: relax one LLM constraint and retry (base never relaxed).
    if len(conditions) > 1:
        relaxed = _drop_lowest_confidence(conditions)
        results = _execute(base + relaxed, db)
        if len(results) >= 3:
            return results

    return None  # semantic-only fallback
