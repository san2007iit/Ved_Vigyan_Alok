"""
crm_tools.py
------------
Tools for reading a customer's record: profile, order history, ticket
history, and a computed set of customer KPIs.

Finance edition -> customer-care edition:
  get_income_statement     -> get_customer_profile   (who they are, spend and margin by year)
  get_balance_sheet        -> get_order_history      (what they own with us, returns, warranties)
  get_cash_flow_statement  -> get_ticket_history     (how service has flowed: tickets, speed, CSAT)
  compute_key_ratios       -> compute_customer_kpis  (contact rate, FCR, CSAT, return rate...)

All tools use the @function_tool decorator from the OpenAI Agents SDK and
return JSON strings. The numbers are computed in Python, never by the LLM.
"""

import json
from datetime import date

from agents import function_tool

from tools import cache
from tools.mock_crm import AS_OF, CUSTOMERS, ORDERS, TICKETS, lookup


def _cached(name: str, customer_id: str, build) -> str:
    key = (name, customer_id.strip().upper())
    hit = cache.get(key)
    if hit is not None:
        return hit
    try:
        value = json.dumps(build(), ensure_ascii=False)
    except KeyError as exc:
        value = json.dumps({"error": str(exc).strip("'\""), "customer_id": customer_id})
    cache.store(key, value)
    return value


def _days_between(a: str, b: date) -> int:
    return (b - date.fromisoformat(a)).days


# ---------------------------------------------------------------------------
# Pure implementations (easy to unit-test without the Agents SDK)
# ---------------------------------------------------------------------------

def customer_profile(customer_id: str) -> dict:
    c = lookup(CUSTOMERS, customer_id)
    tenure = round(_days_between(c["member_since"], AS_OF) / 365.25, 1)
    spend = c["annual_spend_inr"]
    growth = None
    if spend.get("2024"):
        growth = round((spend["2025"] - spend["2024"]) / spend["2024"] * 100, 1)
    return {
        "customer_id": customer_id.strip().upper(), "name": c["name"], "tier": c["tier"],
        "city": c["city"], "region": c["region"], "tenure_years": tenure,
        "preferred_channel": c["preferred_channel"], "language": c["language"],
        "annual_spend_inr": spend, "annual_margin_inr": c["annual_margin_inr"],
        "spend_growth_2025_vs_2024_pct": growth,
        "trailing_12m_margin_inr": c["trailing_12m_margin_inr"],
        "lifetime_spend_inr": c["lifetime_spend_inr"], "lifetime_orders": c["lifetime_orders"],
        "as_of": AS_OF.isoformat(),
    }


def order_history(customer_id: str) -> dict:
    orders = lookup(ORDERS, customer_id)
    open_orders = [o["order_id"] for o in orders if o["status"] in ("IN_TRANSIT", "OUT_FOR_DELIVERY")]
    returned = [o for o in orders if o["returned"]]
    under_warranty = [o["order_id"] for o in orders
                      if o.get("warranty_until") and date.fromisoformat(o["warranty_until"]) >= AS_OF]
    return {
        "customer_id": customer_id.strip().upper(),
        "orders": orders,
        "summary": {
            "orders_listed": len(orders),
            "open_orders": open_orders,
            "returned_orders": len(returned),
            "returned_value_inr": sum(o["amount_inr"] for o in returned),
            "orders_under_warranty": under_warranty,
            "orders_with_failed_repairs": [o["order_id"] for o in orders if o.get("repair_attempts", 0) >= 2],
        },
        "as_of": AS_OF.isoformat(),
    }


def ticket_history(customer_id: str) -> dict:
    tickets = lookup(TICKETS, customer_id)
    last_90 = [t for t in tickets if _days_between(t["opened"], AS_OF) <= 90]
    return {
        "customer_id": customer_id.strip().upper(),
        "tickets": tickets,
        "summary": {
            "tickets_total": len(tickets),
            "tickets_last_90_days": len(last_90),
            "open_or_escalated": [t["ticket_id"] for t in tickets if t["status"] != "RESOLVED"],
            "escalations": sum(1 for t in tickets if t["escalated"]),
            "lowest_recent_csat": min((t["csat"] for t in last_90 if t["csat"] is not None), default=None),
        },
        "as_of": AS_OF.isoformat(),
    }


def customer_kpis(customer_id: str) -> dict:
    c = lookup(CUSTOMERS, customer_id)
    orders = lookup(ORDERS, customer_id)
    tickets = lookup(TICKETS, customer_id)
    orders_12m = [o for o in orders if _days_between(o["date"], AS_OF) <= 365]
    tickets_12m = [t for t in tickets if _days_between(t["opened"], AS_OF) <= 365]
    contacts_12m = sum(t["contacts"] for t in tickets_12m)
    resolved = [t for t in tickets_12m if t["status"] == "RESOLVED"]
    csats = [t["csat"] for t in tickets_12m if t["csat"] is not None]

    def ratio(n, d, digits=2):
        return round(n / d, digits) if d else None

    kpis = {
        "contacts_per_order_12m": ratio(contacts_12m, len(orders_12m)),
        "repeat_contact_rate_12m": ratio(sum(1 for t in tickets_12m if t["contacts"] > 1), len(tickets_12m)),
        "first_contact_resolution_12m": ratio(sum(1 for t in resolved if t["contacts"] == 1), len(tickets_12m)),
        "avg_resolution_hours_12m": ratio(sum(t["resolution_hours"] for t in resolved), len(resolved), 1),
        "avg_first_response_hours_12m": ratio(sum(t["first_response_hours"] for t in tickets_12m), len(tickets_12m), 1),
        "csat_avg_12m": ratio(sum(csats), len(csats)),
        "escalation_rate_12m": ratio(sum(1 for t in tickets_12m if t["escalated"]), len(tickets_12m)),
        "return_rate_12m": ratio(sum(1 for o in orders_12m if o["returned"]), len(orders_12m)),
        "margin_pct_2025": ratio(c["annual_margin_inr"]["2025"] * 100, c["annual_spend_inr"]["2025"], 1),
        "days_since_last_order": min(_days_between(o["date"], AS_OF) for o in orders),
        "oldest_open_ticket_days": max((_days_between(t["opened"], AS_OF) for t in tickets
                                        if t["status"] != "RESOLVED"), default=0),
    }
    guide = {
        "contacts_per_order_12m": "healthy <= 0.40; > 1.0 means the customer needs us far too often",
        "repeat_contact_rate_12m": "healthy <= 0.15; > 0.30 signals unresolved issues",
        "first_contact_resolution_12m": "healthy >= 0.70; < 0.50 is poor",
        "avg_resolution_hours_12m": "healthy <= 24; > 72 is poor",
        "csat_avg_12m": "1-5 scale; healthy >= 4.2; < 3.0 is poor",
        "escalation_rate_12m": "healthy <= 0.05; > 0.25 is critical",
        "return_rate_12m": "healthy <= 0.10; > 0.40 warrants an abuse review",
        "margin_pct_2025": "NovaMart average about 18%; negative means we lose money serving the customer",
        "oldest_open_ticket_days": "healthy <= 3; > 14 means an issue has been open far too long",
    }
    return {"customer_id": customer_id.strip().upper(), "tier": c["tier"], "kpis": kpis,
            "interpretation_guide": guide, "as_of": AS_OF.isoformat()}


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

@function_tool
def get_customer_profile(customer_id: str) -> str:
    """
    Get a NovaMart customer's profile: tier, tenure, region, preferred channel,
    language, annual spend and margin by year, lifetime spend. Returns JSON.

    Args:
        customer_id: The customer ID, e.g. 'C1001'.
    """
    return _cached("profile", customer_id, lambda: customer_profile(customer_id))


@function_tool
def get_order_history(customer_id: str) -> str:
    """
    Get a customer's order history: products, amounts, delivery status,
    returns, warranties and repair attempts. Returns JSON.

    Args:
        customer_id: The customer ID, e.g. 'C1001'.
    """
    return _cached("orders", customer_id, lambda: order_history(customer_id))


@function_tool
def get_ticket_history(customer_id: str) -> str:
    """
    Get a customer's support-ticket history: channel, category, first-response
    and resolution times, status, contacts per ticket, CSAT and escalations. Returns JSON.

    Args:
        customer_id: The customer ID, e.g. 'C1001'.
    """
    return _cached("tickets", customer_id, lambda: ticket_history(customer_id))


@function_tool
def compute_customer_kpis(customer_id: str) -> str:
    """
    Compute a customer's service and value KPIs over the last 12 months:
    contacts per order, repeat-contact rate, first-contact resolution,
    resolution time, CSAT, escalation rate, return rate, margin %.
    Returns JSON with an interpretation guide for each KPI.

    Args:
        customer_id: The customer ID, e.g. 'C1001'.
    """
    return _cached("kpis", customer_id, lambda: customer_kpis(customer_id))
