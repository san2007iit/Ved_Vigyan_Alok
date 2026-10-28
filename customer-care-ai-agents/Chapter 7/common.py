"""
common.py - Shared configuration and helpers for the Chapter 7 multi-agent labs
(customer-care edition).

The task in every lab: assess the HEALTH of a NovaMart B2B service account
(hotel chains, school trusts, clinics... that buy appliances and a service
contract from NovaMart) and decide whether the relationship is healthy.

Finance edition -> customer-care edition:
  profitability ratios (ROA, ROE, margins)   -> SERVICE QUALITY metrics (CSAT, FCR, SLA, self-service)
  liquidity ratios (current, quick, D/E...)  -> RELATIONSHIP RISK metrics (repeat contacts,
                                                escalations, first response, renewal probability)
  stock price / company info / income stmt   -> SLA report / account profile / billing summary

The LlamaIndex and OpenAI Agents SDK labs share these definitions so the
*task* stays identical and you can focus on the *architecture* each
framework expresses. The AutoGen lab analyses a ticket CSV with generated code.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Account-health thresholds - used by the Service Quality and Relationship
# Risk agents to score each metric on a 1-10 scale.
# "unit" tells the prompt renderer how to display the numbers.
# ---------------------------------------------------------------------------

SERVICE_QUALITY_THRESHOLDS = {
    "CSAT": {
        "healthy": 0.85, "moderate": 0.75, "unit": "pct",
        "description": "Share of survey responses rated 4 or 5 out of 5",
    },
    "First Contact Resolution": {
        "healthy": 0.70, "moderate": 0.60, "unit": "pct",
        "description": "Share of issues solved in the first contact",
    },
    "SLA Compliance": {
        "healthy": 0.95, "moderate": 0.90, "unit": "pct",
        "description": "Share of tickets meeting the contractual response and resolution SLA",
    },
    "Self-Service Resolution": {
        "healthy": 0.40, "moderate": 0.25, "unit": "pct",
        "description": "Share of issues solved through the portal or bot without an agent",
    },
}

RELATIONSHIP_RISK_THRESHOLDS = {
    "Repeat Contact Rate": {
        "healthy_max": 0.15, "warning": 0.25, "unit": "pct",
        "description": "Share of issues needing more than one contact (lower is better)",
    },
    "Escalation Rate": {
        "healthy_max": 0.05, "warning": 0.10, "unit": "pct",
        "description": "Share of tickets escalated to a manager or the account team (lower is better)",
    },
    "Avg First Response Hours": {
        "healthy_max": 4.0, "warning": 8.0, "unit": "hours",
        "description": "Average time to the first human response (lower is better)",
    },
    "Renewal Probability": {
        "healthy": 0.80, "moderate": 0.60, "unit": "pct",
        "description": "Customer-success manager's estimated probability that the contract renews",
    },
}


def format_threshold_prompt(thresholds: dict, category: str) -> str:
    """Render a threshold dict into a prompt snippet for an LLM agent."""
    lines = [f"## {category} Thresholds\n"]
    for metric, config in thresholds.items():
        unit = config.get("unit", "pct")
        lines.append(f"### {metric}")
        lines.append(f"Description: {config['description']}")
        for key, value in config.items():
            if key in ("description", "unit"):
                continue
            # Rates are shown as percentages; hours stay plain numbers (do not render 4.0 as 400%).
            lines.append(f"  {key}: {value:.0%}" if unit == "pct" else f"  {key}: {value} h")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stub account data - keeps the labs reproducible without a CRM connection.
# Real implementations would call the CRM, ticketing and billing APIs.
# ---------------------------------------------------------------------------

STUB_ACCOUNTS = {
    "AC-HOTELS": {
        "service_quality": {"CSAT": 0.78, "First Contact Resolution": 0.62,
                            "SLA Compliance": 0.88, "Self-Service Resolution": 0.18},
        "relationship_risk": {"Repeat Contact Rate": 0.27, "Escalation Rate": 0.12,
                              "Avg First Response Hours": 9.5, "Renewal Probability": 0.55},
        "sla_report": {"account_id": "AC-HOTELS", "period": "Q3 2026", "tickets": 412, "open_p1": 3,
                       "response_sla_met": 0.91, "resolution_sla_met": 0.84, "sla_breaches": 49,
                       "top_issue": "AC cooling failures during peak occupancy"},
        "profile": {"company_name": "Sunrise Hotels Group", "industry": "Hospitality", "sites": 42,
                    "segment": "Enterprise", "annual_contract_value_inr": 18_000_000,
                    "renewal_date": "2027-01-31", "customer_success_manager": "Neha Kapoor"},
        "billing": {"invoices_open": 6, "overdue_amount_inr": 1_240_000, "days_overdue_max": 47,
                    "service_credits_issued_ytd_inr": 380_000, "last_payment": "2026-09-30",
                    "dispute": "Customer withholding payment pending SLA-credit review"},
    },
    "AC-SCHOOLS": {
        "service_quality": {"CSAT": 0.91, "First Contact Resolution": 0.79,
                            "SLA Compliance": 0.97, "Self-Service Resolution": 0.46},
        "relationship_risk": {"Repeat Contact Rate": 0.09, "Escalation Rate": 0.03,
                              "Avg First Response Hours": 2.5, "Renewal Probability": 0.90},
        "sla_report": {"account_id": "AC-SCHOOLS", "period": "Q3 2026", "tickets": 168, "open_p1": 0,
                       "response_sla_met": 0.98, "resolution_sla_met": 0.96, "sla_breaches": 5,
                       "top_issue": "Projector lamp replacements"},
        "profile": {"company_name": "Vidya Schools Trust", "industry": "Education", "sites": 18,
                    "segment": "Mid-market", "annual_contract_value_inr": 4_500_000,
                    "renewal_date": "2027-04-30", "customer_success_manager": "Arvind Rao"},
        "billing": {"invoices_open": 1, "overdue_amount_inr": 0, "days_overdue_max": 0,
                    "service_credits_issued_ytd_inr": 15_000, "last_payment": "2026-10-15", "dispute": None},
    },
    "AC-CLINICS": {
        "service_quality": {"CSAT": 0.84, "First Contact Resolution": 0.71,
                            "SLA Compliance": 0.93, "Self-Service Resolution": 0.31},
        "relationship_risk": {"Repeat Contact Rate": 0.16, "Escalation Rate": 0.06,
                              "Avg First Response Hours": 4.5, "Renewal Probability": 0.78},
        "sla_report": {"account_id": "AC-CLINICS", "period": "Q3 2026", "tickets": 236, "open_p1": 1,
                       "response_sla_met": 0.95, "resolution_sla_met": 0.91, "sla_breaches": 17,
                       "top_issue": "Medical-grade refrigerator temperature alarms"},
        "profile": {"company_name": "CarePlus Clinics", "industry": "Healthcare", "sites": 27,
                    "segment": "Enterprise", "annual_contract_value_inr": 9_600_000,
                    "renewal_date": "2026-12-31", "customer_success_manager": "Farah Siddiqui"},
        "billing": {"invoices_open": 2, "overdue_amount_inr": 210_000, "days_overdue_max": 12,
                    "service_credits_issued_ytd_inr": 90_000, "last_payment": "2026-10-02", "dispute": None},
    },
    "AC-COWORK": {
        "service_quality": {"CSAT": 0.88, "First Contact Resolution": 0.74,
                            "SLA Compliance": 0.96, "Self-Service Resolution": 0.52},
        "relationship_risk": {"Repeat Contact Rate": 0.12, "Escalation Rate": 0.04,
                              "Avg First Response Hours": 3.0, "Renewal Probability": 0.62},
        "sla_report": {"account_id": "AC-COWORK", "period": "Q3 2026", "tickets": 140, "open_p1": 0,
                       "response_sla_met": 0.97, "resolution_sla_met": 0.95, "sla_breaches": 7,
                       "top_issue": "Coffee-machine and smart-lock servicing"},
        "profile": {"company_name": "BlueOak Coworking", "industry": "Real estate", "sites": 11,
                    "segment": "Mid-market", "annual_contract_value_inr": 3_200_000,
                    "renewal_date": "2026-12-15", "customer_success_manager": "Rohan Mehta"},
        "billing": {"invoices_open": 1, "overdue_amount_inr": 0, "days_overdue_max": 0,
                    "service_credits_issued_ytd_inr": 0, "last_payment": "2026-10-20",
                    "dispute": "Procurement asked for a 15% price cut at renewal (budget pressure)"},
    },
    "AC-RETAIL": {
        "service_quality": {"CSAT": 0.69, "First Contact Resolution": 0.55,
                            "SLA Compliance": 0.81, "Self-Service Resolution": 0.12},
        "relationship_risk": {"Repeat Contact Rate": 0.31, "Escalation Rate": 0.14,
                              "Avg First Response Hours": 11.0, "Renewal Probability": 0.40},
        "sla_report": {"account_id": "AC-RETAIL", "period": "Q3 2026", "tickets": 538, "open_p1": 5,
                       "response_sla_met": 0.86, "resolution_sla_met": 0.77, "sla_breaches": 124,
                       "top_issue": "Cold-room compressor failures in summer"},
        "profile": {"company_name": "FreshBasket Stores", "industry": "Grocery retail", "sites": 64,
                    "segment": "Enterprise", "annual_contract_value_inr": 22_000_000,
                    "renewal_date": "2026-11-30", "customer_success_manager": "Imran Shaikh"},
        "billing": {"invoices_open": 9, "overdue_amount_inr": 3_150_000, "days_overdue_max": 68,
                    "service_credits_issued_ytd_inr": 1_120_000, "last_payment": "2026-08-28",
                    "dispute": "Escalated to the customer's COO; competitor proposal received"},
    },
}


def stub_lookup(account_id: str) -> dict:
    """Return the canned data block for an account, raising on unknown accounts."""
    a = account_id.strip().upper()
    if a not in STUB_ACCOUNTS:
        raise ValueError(f"No stub data for {a}. Add it to STUB_ACCOUNTS or wire in the real CRM.")
    return STUB_ACCOUNTS[a]
