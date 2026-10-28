"""Tree of Thoughts (Yao et al., 2023) - customer-care edition.

Choosing a SERVICE-RECOVERY strategy for a high-value customer whose case
has gone badly wrong. The search is a two-level tree with beam search:

  Level 1 - propose distinct recovery strategies; score each on customer
            impact, cost and policy/precedent risk; keep the top beam_width.
  Level 2 - for each retained strategy, propose distinct implementations
            (credit amount, timeline, who calls, what is promised); score each;
            keep the best.

Finance edition: hedging a concentrated NVDA position into earnings
(level 1 = hedge instrument, level 2 = strike / notional / expiry).
"""
import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = os.getenv("CC_ANTHROPIC_MODEL", "claude-opus-5-5")

CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


def ask_json(prompt: str, schema: dict) -> dict:
    """One Claude call whose answer is guaranteed to match `schema` (structured outputs)."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request: {response.stop_details}")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def generate_strategies(case: str, situation: str, n: int) -> list[dict]:
    prompt = (f"Customer case: {case}\nSituation: {situation}\n"
              f"Propose {n} distinct service-recovery strategies, each with a short rationale. "
              f"Make them materially different (e.g. speed, money, product, relationship).")
    return ask_json(prompt, CANDIDATES_SCHEMA)["candidates"]


def score_strategy(case: str, situation: str, strategy: dict) -> dict:
    prompt = (f"Customer case: {case}\nSituation: {situation}\n"
              f"Candidate strategy: {strategy['name']} ({strategy['rationale']})\n"
              f"Score this strategy from 1 to 10 on: likelihood of retaining the customer and "
              f"defusing the public/legal threat, cost to NovaMart, and policy and precedent risk.")
    result = ask_json(prompt, SCORE_SCHEMA)
    print(json.dumps(result))
    return result


def generate_implementations(case: str, situation: str, strategy: dict, n: int) -> list[dict]:
    prompt = (f"Customer case: {case}\nSituation: {situation}\n"
              f"Parent strategy: {strategy['name']} ({strategy['rationale']})\n"
              f"Propose {n} distinct implementations of this strategy (goodwill amount, timeline, "
              f"who contacts the customer and how, exactly what is promised), each with a short rationale.")
    return ask_json(prompt, CANDIDATES_SCHEMA)["candidates"]


def score_implementation(case: str, situation: str, strategy: dict, impl: dict) -> dict:
    prompt = (f"Customer case: {case}\nSituation: {situation}\n"
              f"Parent strategy: {strategy['name']} ({strategy['rationale']})\n"
              f"Candidate implementation: {impl['name']} ({impl['rationale']})\n"
              f"Score this implementation from 1 to 10 on residual churn and escalation risk, cost, "
              f"compliance with approval limits, and how cleanly it delivers the parent strategy.")
    return ask_json(prompt, SCORE_SCHEMA)


def best_recovery(case: str, situation: str, beam_width: int = 2) -> list[dict]:
    # Level 1: pick the best strategies
    candidates = generate_strategies(case, situation, n=4)
    print("\n\nCANDIDATE STRATEGIES\n" + "=" * 20)
    print(json.dumps(candidates, indent=2))
    scored = [{**c, **score_strategy(case, situation, c)} for c in candidates]
    scored.sort(key=lambda x: x["score"], reverse=True)
    print("\n\nSCORED\n" + "=" * 20)
    print(json.dumps(scored, indent=2))

    retained = scored[:beam_width]
    print("\n\nRETAINED (beam)\n" + "=" * 20)
    print(json.dumps([r["name"] for r in retained], indent=2))

    # Level 2: for each retained strategy, pick its best implementation
    for strategy in retained:
        impls = generate_implementations(case, situation, strategy, n=3)
        scored_impls = [{**i, **score_implementation(case, situation, strategy, i)} for i in impls]
        scored_impls.sort(key=lambda x: x["score"], reverse=True)
        print(f"\n\nIMPLEMENTATIONS FOR: {strategy['name']}\n" + "=" * 20)
        print(json.dumps(scored_impls, indent=2))
        strategy["best_implementation"] = scored_impls[0]
    return retained


if __name__ == "__main__":
    case = ("Priya Sharma (C1001), Platinum member for 7 years, lifetime spend Rs 4,85,000. FrostLine "
            "refrigerator (Rs 62,990) failed twice after repair; part back-ordered until 15 November. "
            "Policy: replacement under the repeat-failure rule; goodwill up to Rs 1,000 (agent), Rs 5,000 "
            "(supervisor), above that a CX manager. Estimated customer lifetime value: Rs 90,000.")
    situation = ("She posted on Twitter (1.2k reposts) and will send a consumer-court notice if the issue is "
                 "not fixed by Thursday. A competitor just launched a 7-day replacement guarantee.")
    for s in best_recovery(case, situation):
        print(f"{s['score']:>2}  {s['name']}: {s['reasoning']}")
        b = s["best_implementation"]
        print(f"      +-- best implementation ({b['score']}): {b['name']}: {b['reasoning']}")
