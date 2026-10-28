"""
voc_tools.py
------------
Voice-of-the-customer tools: retrieve what the customer has said across
channels and score its sentiment and risk flags.

Finance edition -> customer-care edition:
  search_recent_news  -> search_customer_interactions  (messages, call notes, social posts, reviews)
  analyze_sentiment   -> analyze_sentiment             (aggregate score, themes, MATERIAL RISK FLAGS)

The preliminary sentiment is a transparent keyword score, so students can
see exactly why a message was labelled negative. In production you would
use a sentiment model; the agents would not change.
"""

import json

from agents import function_tool

from tools import cache
from tools.mock_crm import AS_OF, INTERACTIONS, lookup

_POSITIVE = ["thanks", "thank you", "helpful", "great", "polite", "quick", "happy", "easy", "trusted", "resolved"]
_NEGATIVE = ["unacceptable", "disappointed", "failed", "worst", "cracked", "frustrated", "late", "delay",
             "not moved", "without a working", "threw away", "worried", "blocked", "immediately", "#novamartfail"]

# Material risk flags: each one changes how the case must be handled.
RISK_FLAGS = {
    "SAFETY": ["smoke", "burning", "fire", "spark", "shock", "gas leak", "injury"],
    "LEGAL_THREAT": ["consumer court", "legal notice", "lawyer", "consumer forum"],
    "CHURN_SIGNAL": ["another store", "switch to", "never again", "move to", "cancel my"],
    "PAYMENT_RISK": ["different card", "chargeback", "new account", "refund to another"],
}


def _score(text: str) -> tuple[str, float]:
    low = text.lower()
    pos = sum(1 for w in _POSITIVE if w in low)
    neg = sum(1 for w in _NEGATIVE if w in low)
    score = round((pos - neg) / max(1, pos + neg), 2)
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return label, score


# Public channels: a NEGATIVE message here is a public complaint (reputational risk).
PUBLIC_SOURCES = ["twitter", "facebook", "instagram", "review", "youtube"]


def _flags(text: str, source: str, sentiment: str) -> list[str]:
    low = text.lower()
    flags = [flag for flag, words in RISK_FLAGS.items() if any(w in low for w in words)]
    if sentiment == "negative" and any(p in source.lower() for p in PUBLIC_SOURCES):
        flags.append("PUBLIC_COMPLAINT")
    return flags


def interactions(customer_id: str, max_results: int = 10) -> dict:
    items = lookup(INTERACTIONS, customer_id)
    items = sorted(items, key=lambda i: i["date"], reverse=True)[:max_results]
    enriched = []
    for it in items:
        label, score = _score(it["text"])
        enriched.append({**it, "sentiment": label, "score": score, "flags": _flags(it["text"], it["source"], label)})
    return {"customer_id": customer_id.strip().upper(), "count": len(enriched),
            "interactions": enriched, "as_of": AS_OF.isoformat()}


def sentiment_summary(interactions_json: str) -> dict:
    data = json.loads(interactions_json)
    items = data.get("interactions", [])
    if not items:
        return {"error": "No interactions to analyse"}
    # Recent messages matter more: weight 1.0 for the latest, decaying by 15% per older message.
    weights = [0.85 ** i for i in range(len(items))]
    agg = round(sum(w * it["score"] for w, it in zip(weights, items)) / sum(weights), 2)
    verdict = ("VERY NEGATIVE" if agg <= -0.6 else "NEGATIVE" if agg <= -0.2
               else "NEUTRAL" if agg < 0.2 else "POSITIVE" if agg < 0.6 else "VERY POSITIVE")
    flags = sorted({f for it in items for f in it.get("flags", [])})
    channels = sorted({it["source"] for it in items})
    return {
        "customer_id": data.get("customer_id"), "aggregate_score": agg, "sentiment_verdict": verdict,
        "material_risk_flags": flags, "channels": channels,
        "negative_messages": sum(1 for it in items if it["sentiment"] == "negative"),
        "positive_messages": sum(1 for it in items if it["sentiment"] == "positive"),
        "all_interactions": items,
    }


@function_tool
def search_customer_interactions(customer_id: str, max_results: int) -> str:
    """
    Retrieve the customer's most recent interactions across channels
    (WhatsApp, email, call notes, social media, app reviews), each with a
    preliminary sentiment label and any risk flags. Returns JSON.

    Args:
        customer_id: The customer ID, e.g. 'C1001'.
        max_results: Maximum number of interactions to return (e.g. 10).
    """
    key = ("interactions", customer_id.strip().upper(), max_results)
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        value = json.dumps(interactions(customer_id, max_results), ensure_ascii=False)
    except KeyError as exc:
        value = json.dumps({"error": str(exc).strip("'\""), "customer_id": customer_id})
    cache.store(key, value)
    return value


@function_tool
def analyze_sentiment(interactions_json: str) -> str:
    """
    Aggregate the sentiment of a set of customer interactions (the JSON
    returned by search_customer_interactions): recency-weighted score,
    sentiment verdict, material risk flags and channels. Returns JSON.

    Args:
        interactions_json: The JSON string returned by search_customer_interactions.
    """
    try:
        return json.dumps(sentiment_summary(interactions_json), ensure_ascii=False)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return json.dumps({"error": f"Could not parse interactions: {exc}"})
