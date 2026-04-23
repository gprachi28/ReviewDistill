"""
benchmarks/query_eval.py

Manual evaluation set: 20 queries covering all 3 intents.

Two modes:
    --mode filter   (default) — run expected_filters through filter_businesses()
                                 to verify DB coverage. No LLM required.
    --mode planner  — call plan_query() for each case, check intent + filters.
                       Requires vllm server running at settings.vllm_base_url.

Usage:
    python benchmarks/query_eval.py
    python benchmarks/query_eval.py --mode planner
"""
import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, ".")

from api.schemas import Intent
from api.sql_filter import filter_businesses


@dataclass
class EvalCase:
    id: str
    question: str
    expected_intent: Intent
    # Filters passed to filter_businesses() in filter mode; ground truth in planner mode.
    expected_filters: dict[str, Any]
    # Keys that plan_query() MUST produce (planner mode only).
    required_filter_keys: list[str]
    # Keys that MUST NOT appear in the plan — hallucination guard (planner mode only).
    forbidden_filter_keys: list[str] = field(default_factory=list)
    notes: str = ""


EVAL_CASES: list[EvalCase] = [
    # ── find_businesses (14) ─────────────────────────────────────────
    EvalCase(
        id="FB-01",
        question="bachelor party spot, loud, handles large groups",
        expected_intent="find_businesses",
        expected_filters={"noise_level": ["loud", "very_loud"], "good_for_groups": True},
        required_filter_keys=["noise_level", "good_for_groups"],
        forbidden_filter_keys=["attire", "price_range"],
        notes="EXP-001: attire was hallucinated in v1 prompt",
    ),
    EvalCase(
        id="FB-02",
        question="cheap brunch place with outdoor seating",
        expected_intent="find_businesses",
        expected_filters={"good_for_meal": {"brunch": True}, "outdoor_seating": True, "price_range": {"lte": 2}},
        required_filter_keys=["good_for_meal", "outdoor_seating", "price_range"],
    ),
    EvalCase(
        id="FB-03",
        question="jazz brunch with live music",
        expected_intent="find_businesses",
        expected_filters={"good_for_meal": {"brunch": True}, "music": {"live": True}},
        required_filter_keys=["good_for_meal", "music"],
    ),
    EvalCase(
        id="FB-04",
        question="late-night Cajun food after a show on Frenchmen Street",
        expected_intent="find_businesses",
        expected_filters={"good_for_meal": {"latenight": True}},
        required_filter_keys=["good_for_meal"],
    ),
    EvalCase(
        id="FB-05",
        question="family-friendly seafood place with parking",
        expected_intent="find_businesses",
        expected_filters={"good_for_kids": True},
        required_filter_keys=["good_for_kids"],
    ),
    EvalCase(
        id="FB-06",
        question="quiet romantic date spot",
        expected_intent="find_businesses",
        expected_filters={"noise_level": "quiet", "ambience": {"romantic": True}},
        required_filter_keys=["noise_level", "ambience"],
    ),
    EvalCase(
        id="FB-07",
        question="outdoor patio restaurant with a full bar",
        expected_intent="find_businesses",
        expected_filters={"outdoor_seating": True, "alcohol": "full_bar"},
        required_filter_keys=["outdoor_seating", "alcohol"],
    ),
    EvalCase(
        id="FB-08",
        question="dog-friendly restaurant with a patio",
        expected_intent="find_businesses",
        expected_filters={"dogs_allowed": True, "outdoor_seating": True},
        required_filter_keys=["dogs_allowed", "outdoor_seating"],
    ),
    EvalCase(
        id="FB-09",
        question="upscale dinner spot that takes reservations",
        expected_intent="find_businesses",
        expected_filters={"ambience": {"upscale": True}, "takes_reservations": True},
        required_filter_keys=["ambience", "takes_reservations"],
    ),
    EvalCase(
        id="FB-10",
        question="happy hour bar with TVs to watch sports",
        expected_intent="find_businesses",
        expected_filters={"happy_hour": True, "has_tv": True},
        required_filter_keys=["happy_hour", "has_tv"],
    ),
    EvalCase(
        id="FB-11",
        question="BYOB place with casual attire",
        expected_intent="find_businesses",
        expected_filters={"byob": True, "attire": "casual"},
        required_filter_keys=["byob", "attire"],
    ),
    EvalCase(
        id="FB-12",
        question="bachelorette dinner, upscale vibes, good for large groups",
        expected_intent="find_businesses",
        expected_filters={"ambience": {"upscale": True}, "good_for_groups": True},
        required_filter_keys=["good_for_groups"],
        forbidden_filter_keys=["attire", "noise_level"],
        notes="bachelorette should not imply attire or noise_level",
    ),
    EvalCase(
        id="FB-13",
        question="wheelchair accessible restaurant that caters events",
        expected_intent="find_businesses",
        expected_filters={"wheelchair_accessible": True, "caters": True},
        required_filter_keys=["wheelchair_accessible", "caters"],
    ),
    EvalCase(
        id="FB-14",
        question="dressy dinner spot with live music",
        expected_intent="find_businesses",
        expected_filters={"attire": "dressy", "music": {"live": True}},
        required_filter_keys=["attire", "music"],
    ),
    # ── question_about_business (4) ───────────────────────────────────
    EvalCase(
        id="QB-01",
        question="What do people say about parking at Acme Oyster House?",
        expected_intent="question_about_business",
        expected_filters={},
        required_filter_keys=[],
    ),
    EvalCase(
        id="QB-02",
        question="Is Jacques-Imo's good for a large group?",
        expected_intent="question_about_business",
        expected_filters={},
        required_filter_keys=[],
    ),
    EvalCase(
        id="QB-03",
        question="What's the vibe like at The Maison on Frenchmen Street?",
        expected_intent="question_about_business",
        expected_filters={},
        required_filter_keys=[],
    ),
    EvalCase(
        id="QB-04",
        question="Do people recommend Dooky Chase's for a special occasion?",
        expected_intent="question_about_business",
        expected_filters={},
        required_filter_keys=[],
    ),
    # ── compare_businesses (2) ────────────────────────────────────────
    EvalCase(
        id="CB-01",
        question="How does Jacques-Imo's compare to La Casita for a birthday party?",
        expected_intent="compare_businesses",
        expected_filters={},
        required_filter_keys=[],
    ),
    EvalCase(
        id="CB-02",
        question="Is Galatoire's or Antoine's better for a romantic anniversary dinner?",
        expected_intent="compare_businesses",
        expected_filters={},
        required_filter_keys=[],
    ),
]


# ── filter mode ──────────────────────────────────────────────────────────────


def run_filter_mode() -> None:
    """Verify expected_filters return ≥3 businesses from the DB. No LLM needed."""
    print("FILTER MODE — DB coverage check (no LLM required)")
    print("─" * 60)

    passed = 0
    skipped = 0
    fb_total = 0

    for case in EVAL_CASES:
        if not case.expected_filters:
            print(f"{case.id:<8} SKIP  (no SQL filters — intent-only case)")
            skipped += 1
            continue

        fb_total += 1
        result = filter_businesses(case.expected_filters)
        count = len(result) if result else 0
        ok = count >= 3
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        question_short = case.question[:55] + "..." if len(case.question) > 55 else case.question
        print(f"{case.id:<8} {status}  {count:>3} businesses  {question_short}")
        if not ok and case.notes:
            print(f"         note: {case.notes}")

    print("─" * 60)
    print(f"Filter accuracy: {passed}/{fb_total}  ({100*passed/fb_total:.1f}%)  [{skipped} QB/CB cases skipped]")


# ── planner mode ─────────────────────────────────────────────────────────────


def _check_filters(produced: dict, required: list[str], forbidden: list[str]) -> tuple[bool, str]:
    """
    Returns (pass, detail_message).
    Pass requires all required keys present and no forbidden keys present.
    """
    missing = [k for k in required if k not in produced]
    present_forbidden = [k for k in forbidden if k in produced]
    if not missing and not present_forbidden:
        return True, "ok"
    parts = []
    if missing:
        parts.append(f"missing={missing}")
    if present_forbidden:
        parts.append(f"forbidden_present={present_forbidden}")
    return False, "  ".join(parts)


def run_planner_mode() -> None:
    """Call plan_query() for each case and check intent + filter fields."""
    from api.query_planner import plan_query

    print("PLANNER MODE — requires vllm server")
    print("─" * 60)

    total = 0
    passed = 0

    for case in EVAL_CASES:
        total += 1
        try:
            plan = plan_query(case.question)
        except Exception as exc:
            print(f"{case.id:<8} ERROR  {exc}")
            continue

        intent_ok = plan.intent == case.expected_intent
        filters_ok, filter_detail = _check_filters(
            plan.sql_filters, case.required_filter_keys, case.forbidden_filter_keys
        )
        case_ok = intent_ok and filters_ok
        if case_ok:
            passed += 1

        status = "PASS" if case_ok else "FAIL"
        intent_mark = "✓" if intent_ok else f"✗ got={plan.intent}"
        filter_mark = "✓" if filters_ok else f"✗ {filter_detail}"
        question_short = case.question[:45] + "..." if len(case.question) > 45 else case.question
        print(f"{case.id:<8} {status}  intent:{intent_mark}  filters:{filter_mark}  {question_short}")
        if not case_ok and case.notes:
            print(f"         note: {case.notes}")

    print("─" * 60)
    print(f"Planner accuracy: {passed}/{total}  ({100*passed/total:.1f}%)")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Query eval for Yelp conversational assistant")
    parser.add_argument(
        "--mode",
        choices=["filter", "planner"],
        default="filter",
        help="filter: DB coverage check (no LLM). planner: full plan_query eval (needs vllm).",
    )
    args = parser.parse_args()

    if args.mode == "filter":
        run_filter_mode()
    else:
        run_planner_mode()


if __name__ == "__main__":
    main()
