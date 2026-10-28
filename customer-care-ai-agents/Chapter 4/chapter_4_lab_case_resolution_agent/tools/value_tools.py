"""
value_tools.py
--------------
Tools for estimating what a customer relationship is worth.

Finance edition -> customer-care edition:
  run_dcf_model       -> run_clv_model           (Customer Lifetime Value = a DCF of customer margin)
  get_peer_multiples  -> get_segment_benchmarks  (compare the customer with their loyalty-tier peers)

Why CLV is "a DCF of a customer": a stock is worth the discounted value of
its future free cash flows; a customer is worth the discounted value of the
margin they will generate while they stay. The extra ingredient is the
retention rate: each year the customer survives with probability r.

    CLV = sum_{t=1..5} m0 (1+g)^t  r^t / (1+d)^t   +  discounted terminal value

The verdict is computed deterministically in Python and reported by the
agent as-is, so the LLM explains the number but does not invent it.
"""

import json
import math

from agents import function_tool

from tools.mock_crm import CUSTOMERS, ORDERS, SEGMENT_BENCHMARKS, lookup

# Company-wide "average customer" used as the yardstick for CLV.
COMPANY_BENCHMARK = {"avg_annual_margin_inr": 3200, "baseline_retention": 0.78}
DEFAULT_GROWTH, DEFAULT_DISCOUNT, DEFAULT_TERMINAL_GROWTH = 0.05, 0.12, 0.02

# CLV / company-average CLV  ->  value verdict (5 levels, like the finance verdict scale)
_CLV_VERDICT_THRESHOLDS = [
    (5.0, "STRATEGIC"),
    (2.0, "HIGH VALUE"),
    (0.75, "CORE"),
    (0.10, "LOW VALUE"),
    (float("-inf"), "MARGINAL"),
]

# Weighted value premium vs tier peers (%) -> verdict. Positive = more valuable than peers.
_PEER_VERDICT_THRESHOLDS = [
    (30, float("inf"), "STRATEGIC"),
    (15, 30, "HIGH VALUE"),
    (-15, 15, "CORE"),
    (-30, -15, "LOW VALUE"),
    (float("-inf"), -30, "MARGINAL"),
]

# Weights for the peer comparison (must sum to 1.0). Return rate is inverted:
# a customer who returns less than peers is MORE valuable.
_PEER_WEIGHTS = {"annual_spend": 0.35, "annual_margin": 0.35, "avg_order_value": 0.15, "return_rate": 0.15}


def clv(margin: float, growth: float, discount: float, retention: float, terminal_growth: float) -> dict:
    """Pure CLV calculation: 5 explicit years + a terminal value. Returns the pieces."""
    years = []
    total = 0.0
    for t in range(1, 6):
        m_t = margin * (1 + growth) ** t
        survival = retention ** t
        pv = m_t * survival / (1 + discount) ** t
        total += pv
        years.append({"year": t, "expected_margin_inr": round(m_t * survival), "present_value_inr": round(pv)})
    q = (1 + terminal_growth) * retention / (1 + discount)
    pv_terminal = (margin * (1 + growth) ** 5 * retention ** 5 / (1 + discount) ** 5) * q / (1 - q)
    return {"years": years, "sum_pv_5y_inr": round(total), "pv_terminal_inr": round(pv_terminal),
            "clv_inr": round(total + pv_terminal)}


def company_average_clv() -> int:
    return clv(COMPANY_BENCHMARK["avg_annual_margin_inr"], DEFAULT_GROWTH, DEFAULT_DISCOUNT,
               COMPANY_BENCHMARK["baseline_retention"], DEFAULT_TERMINAL_GROWTH)["clv_inr"]


def clv_verdict(ratio: float) -> str:
    return next(label for floor, label in _CLV_VERDICT_THRESHOLDS if ratio >= floor)


def clv_model(customer_id: str, growth_rate: float, discount_rate: float,
              retention_rate: float, terminal_growth_rate: float) -> dict:
    c = lookup(CUSTOMERS, customer_id)
    rates = (growth_rate, discount_rate, retention_rate, terminal_growth_rate)
    if (not all(math.isfinite(x) for x in rates) or growth_rate <= -1 or discount_rate <= 0
            or not 0 < retention_rate < 1 or terminal_growth_rate > 0.03
            or (1 + terminal_growth_rate) * retention_rate >= 1 + discount_rate):
        return {"error": "Rates must be finite; 0 < retention < 1; discount > 0; terminal growth <= 3%.",
                "customer_id": customer_id}
    base_margin = c["trailing_12m_margin_inr"]
    result = clv(base_margin, growth_rate, discount_rate, retention_rate, terminal_growth_rate)
    avg = company_average_clv()
    ratio = result["clv_inr"] / avg
    return {
        "customer_id": customer_id.strip().upper(), "name": c["name"], "tier": c["tier"],
        "model": "5-year customer-margin projection x retention, discounted, + terminal value",
        "limitations": "Illustrative: trailing 12-month margin is a proxy for future margin; "
                       "retention is a tier baseline, not a fitted churn model.",
        "inputs": {"base_annual_margin_inr": base_margin, "growth_rate_pct": round(growth_rate * 100, 1),
                   "discount_rate_pct": round(discount_rate * 100, 1),
                   "retention_rate_pct": round(retention_rate * 100, 1),
                   "terminal_growth_pct": round(terminal_growth_rate * 100, 1)},
        "projection": result["years"],
        "valuation": {
            "sum_pv_5y_inr": result["sum_pv_5y_inr"], "pv_terminal_inr": result["pv_terminal_inr"],
            "clv_inr": result["clv_inr"], "company_average_clv_inr": avg,
            "clv_vs_company_average": round(ratio, 2),
            "recovery_budget_guideline_inr": max(0, round(0.05 * result["clv_inr"])),
            "verdict": clv_verdict(ratio),
        },
    }


def segment_comparison(customer_id: str) -> dict:
    c = lookup(CUSTOMERS, customer_id)
    orders = lookup(ORDERS, customer_id)
    peer = SEGMENT_BENCHMARKS[c["tier"]]
    recent = [o for o in orders if o["date"] >= "2025-10-28"]
    subject = {
        "annual_spend": c["annual_spend_inr"]["2025"],
        "annual_margin": c["trailing_12m_margin_inr"],
        "avg_order_value": round(c["lifetime_spend_inr"] / c["lifetime_orders"]),
        "return_rate": round(sum(1 for o in recent if o["returned"]) / len(recent), 2) if recent else 0.0,
    }
    peers = {"annual_spend": peer["avg_annual_spend_inr"], "annual_margin": peer["avg_annual_margin_inr"],
             "avg_order_value": peer["avg_order_value_inr"], "return_rate": peer["return_rate"]}
    premiums = {}
    for k in _PEER_WEIGHTS:
        raw = (subject[k] / peers[k] - 1) * 100
        if k == "return_rate":
            raw = -raw                      # lower return rate than peers = more valuable
        premiums[k] = round(max(-100.0, min(100.0, raw)), 1)   # cap at +/-100% so one metric cannot dominate
    weighted = round(sum(premiums[k] * w for k, w in _PEER_WEIGHTS.items()), 1)
    verdict = next(label for lo, hi, label in _PEER_VERDICT_THRESHOLDS if lo <= weighted < hi)
    return {
        "customer_id": customer_id.strip().upper(), "tier": c["tier"],
        "customer": subject, "tier_average": peers, "value_premium_pct": premiums,
        "peer_verdict": {"weighted_value_premium_pct": weighted, "weights": _PEER_WEIGHTS, "verdict": verdict},
    }


@function_tool
def run_clv_model(customer_id: str, growth_rate: float, discount_rate: float,
                  retention_rate: float, terminal_growth_rate: float) -> str:
    """
    Run a Customer Lifetime Value (CLV) model: projects the customer's annual
    margin for 5 years, weights each year by the probability the customer is
    still with us (retention), discounts it, and adds a terminal value.
    Compares the CLV with the company-average customer and returns a verdict.

    Args:
        customer_id: The customer ID, e.g. 'C1001'.
        growth_rate: Expected annual margin growth for the next 5 years (e.g. 0.05 for 5%).
        discount_rate: Cost of capital used to discount future margin (e.g. 0.12 for 12%).
        retention_rate: Probability the customer stays each year (e.g. 0.92).
        terminal_growth_rate: Margin growth after year 5 (e.g. 0.02; maximum 0.03).
    """
    try:
        return json.dumps(clv_model(customer_id, growth_rate, discount_rate, retention_rate, terminal_growth_rate))
    except KeyError as exc:
        return json.dumps({"error": str(exc).strip("'\""), "customer_id": customer_id})


@function_tool
def get_segment_benchmarks(customer_id: str) -> str:
    """
    Compare a customer with the average customer of the same loyalty tier
    (annual spend, annual margin, average order value, return rate) and
    return the weighted value premium and a peer-based verdict.

    Args:
        customer_id: The customer ID, e.g. 'C1001'.
    """
    try:
        return json.dumps(segment_comparison(customer_id))
    except KeyError as exc:
        return json.dumps({"error": str(exc).strip("'\""), "customer_id": customer_id})
