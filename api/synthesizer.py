"""
api/synthesizer.py

LLM call 2: convert candidate businesses + review evidence into a conversational answer.

Public API:
    synthesize(question, snippets, business_meta) -> tuple[str, list[BusinessResult]]
"""
from collections import defaultdict

from openai import OpenAI

from api.schemas import BusinessResult
from config import settings

_client = OpenAI(base_url=settings.vllm_base_url, api_key="not-required")

_SYSTEM_PROMPT = """You are a knowledgeable local guide for New Orleans restaurants.
Answer the user's question using ONLY the review evidence provided.
Be conversational, specific, and direct. Name the restaurants you recommend.
Do not invent details not present in the reviews."""


def _build_evidence_block(business_id: str, meta: dict, top_snippets: list[dict]) -> str:
    name = meta.get("name", business_id)
    stars = meta.get("stars", "?")
    price = "$" * meta["price_range"] if meta.get("price_range") else "unknown price"
    reviews = "\n".join(f'  - "{s["text"]}"' for s in top_snippets)
    return f"{name} ({stars}★, {price})\n{reviews}"


def _build_user_prompt(question: str, evidence_blocks: list[str]) -> str:
    evidence = "\n\n".join(evidence_blocks)
    return f"Question: {question}\n\nReview evidence:\n{evidence}\n\nAnswer:"


def synthesize(
    question: str,
    snippets: list[dict],
    business_meta: dict[str, dict],
    snippets_per_business: int = 3,
) -> tuple[str, list[BusinessResult]]:
    """
    Generate a conversational answer grounded in review evidence.

    Args:
        question:              The user's original question.
        snippets:              Ranked review snippets from the retriever
                               [{business_id, text, stars, distance}, ...].
        business_meta:         business_id → {name, stars, price_range} from SQLite.
        snippets_per_business: How many top snippets to include per business in the prompt.

    Returns:
        (answer, businesses) where businesses is a list of BusinessResult with the
        top-matching review snippet as evidence.
    """
    # Group snippets by business, preserving retriever ranking order.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for s in snippets:
        bid = s["business_id"]
        if bid in business_meta:
            grouped[bid].append(s)

    if not grouped:
        return "I couldn't find relevant reviews for your query.", []

    # Build evidence blocks for the prompt (top N snippets per business).
    evidence_blocks = []
    for bid, snips in grouped.items():
        meta = business_meta[bid]
        top = snips[:snippets_per_business]
        evidence_blocks.append(_build_evidence_block(bid, meta, top))

    user_prompt = _build_user_prompt(question, evidence_blocks)
    response = _client.chat.completions.create(
        model=settings.vllm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    answer = response.choices[0].message.content.strip()

    # Build BusinessResult list — one per business that has snippets.
    # Evidence = the single best-matching snippet (lowest distance = index 0).
    businesses = []
    for bid, snips in grouped.items():
        meta = business_meta[bid]
        businesses.append(BusinessResult(
            business_id=bid,
            name=meta.get("name", bid),
            stars=meta.get("stars", 0.0),
            price_range=meta.get("price_range"),
            evidence=snips[0]["text"],
        ))

    return answer, businesses
