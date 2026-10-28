"""Self-Consistency (Wang et al., 2022) - customer-care edition.

Sample several independent chain-of-thought reasoning paths for the same
case, parse each path's final answer, and take the majority (plurality) vote.

Task: for an ambiguous product complaint, recommend REPAIR, REPLACE or REFUND.
Finance edition: BUY / HOLD / SELL for a stock.
"""
import os
from collections import Counter

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = os.getenv("CC_ANTHROPIC_MODEL", "claude-opus-5-5")
N_PATHS = 5
# Note: the finance lab sampled at temperature 0.8 to diversify the paths. Current Claude models
# (Opus 4.7 and later, including the default here) do not accept a temperature setting, so
# diversity comes from the model's default sampling.

SYSTEM_PROMPT = """You are a senior after-sales resolution specialist at NovaMart, an Indian electronics retailer.
Analyse the case and recommend exactly one remedy: REPAIR, REPLACE or REFUND.

Reason step by step. Cover:
1. What the policy allows (and what it does not require)
2. The likely root cause and the chance a repair fixes it
3. Customer impact: time without the product, effort, tier and relationship
4. Cost and precedent for NovaMart
5. Overall balance

Be concise: 1-2 sentences per point.

End your response with exactly one line in this plain format (no markdown, no bold, no asterisks):
RECOMMENDATION: <REPAIR|REPLACE|REFUND>"""

CASE_SUMMARY = """
Customer: Rahul Verma (C1002), Gold member since 2021, lifetime spend Rs 1,20,000
Product: BreezeMax Split AC 1.5T (order ORD-79001), Rs 46,490, installed 25 days ago

Issue:
  - Cooling drops intermittently in the afternoons
  - First technician visit (day 18): low refrigerant found, topped up; suspected slow leak at the evaporator coil
  - Problem returned after 6 days

Options and constraints:
  - Spare evaporator coil available in 3 days; repair takes half a day; leak-repair success rate about 80%
  - Replacement unit: 9 days (stock arriving from the Mumbai warehouse); installation charge Rs 1,500
  - Refund policy: full refund within 30 days of delivery if a manufacturing defect is confirmed
  - Repeat-failure rule (KB-102): replacement is mandatory only after 2 failed repairs (1 so far)

Customer context:
  - Works from home; outdoor temperature 36 C this week
  - Says: "I just want a working AC as fast as possible. I'm losing trust in this brand."
"""


def sample_reasoning_path(case_summary: str) -> str:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": case_summary}],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "RECOMMENDATION: NONE (declined)"
    return "".join(block.text for block in response.content if block.type == "text")


def extract_recommendation(path: str) -> str | None:
    for line in reversed(path.strip().split("\n")):
        cleaned = line.strip().lstrip("*_`# ").upper()
        if cleaned.startswith("RECOMMENDATION:"):
            label = cleaned.replace("RECOMMENDATION:", "").strip().strip("*_`. ")
            if label in ("REPAIR", "REPLACE", "REFUND"):
                return label
    return None


def self_consistency_recommend(case_summary: str, n_paths: int = N_PATHS) -> dict:
    paths, recommendations = [], []
    print(f"[STEP 1] Chain-of-thought system prompt configured. Sampling {n_paths} paths.\n")
    for i in range(1, n_paths + 1):
        print(f"[STEP 2] Sampling reasoning path {i}/{n_paths} ...")
        path = sample_reasoning_path(case_summary)
        recommendation = extract_recommendation(path)
        paths.append(path)
        recommendations.append(recommendation)
        print(f"--- Path {i} reasoning ---\n{path}\n")
        print(f"[Path {i}] Parsed recommendation: {recommendation}\n")

    print("[STEP 3] Aggregating answers via majority vote ...")
    valid = [r for r in recommendations if r is not None]
    if not valid:
        raise RuntimeError("No path produced a parseable recommendation.")
    vote_counts = Counter(valid)
    final, top_votes = vote_counts.most_common(1)[0]
    return {
        "final_recommendation": final,
        "agreement": round(top_votes / len(valid), 2),   # how unanimous the paths were
        "vote_counts": dict(vote_counts),
        "paths": paths,
    }


if __name__ == "__main__":
    result = self_consistency_recommend(CASE_SUMMARY)
    print(f"\n=== FINAL RECOMMENDATION: {result['final_recommendation']} ===")
    print(f"Vote breakdown: {result['vote_counts']}  (agreement {result['agreement']:.0%})")
    if result["agreement"] < 0.6:
        print("Low agreement: route this case to a human supervisor.")
