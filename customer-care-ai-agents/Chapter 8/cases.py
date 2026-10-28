"""
cases.py - synthetic case files for the Chapter 8 escalation committee.

Each case is a POINT-IN-TIME snapshot: only what NovaMart knew when the
decision had to be taken. That is what makes the historical replay in
backtest.py honest (the finance edition's backtest could not do this,
because yfinance returns today's fundamentals, not historical ones).

Historical cases also carry two churn estimates from NovaMart's uplift
model (estimated from past A/B tests of goodwill offers):
  churn_if_standard     - probability of losing the customer with policy-only handling
  churn_if_compensated  - probability with a meaningful goodwill gesture
These are assumptions for teaching, not observed counterfactuals.
"""

OPS_SNAPSHOT = {
    "festive_season": True,
    "queue_backlog_vs_normal_pct": 135,
    "spare_parts_backorders": ["FrostLine relay R-22 (until 15 Nov)"],
    "goodwill_budget_used_this_month_pct": 72,
    "competitor_moves": ["CoolHome: 7-day no-questions replacement guarantee on large appliances"],
}

CASES = {
    "CASE-9031": {
        "customer": "Priya Sharma (C1001)", "tier": "Platinum", "tenure_years": 7.6,
        "lifetime_spend_inr": 485000, "trailing_margin_inr": 15800, "clv_estimate_inr": 90773,
        "order": {"order_id": "ORD-77812", "product": "FrostLine Refrigerator 340L", "value_inr": 62990},
        "issue": "Stopped cooling 3 weeks after delivery; two repairs failed; part back-ordered",
        "company_fault": True,
        "service": {"contacts_last_30d": 9, "avg_contacts_per_month_prior_90d": 0.3,
                    "csat_last_5": [5, 5, 2, 1, 1], "days_open": 19, "failed_fixes": 2},
        "abuse": {"return_rate_12m": 0.0, "damaged_claims_6m": 0, "payment_risk_flag": False},
        "messages": [
            "[2026-10-26, Phone note] Will move to another store and send a consumer court notice if not replaced by Thursday.",
            "[2026-10-25, Twitter] @NovaMartCare 3 weeks without a working fridge after paying Rs 63k #NovaMartFail (1.2k reposts)",
            "[2026-10-20, Email] Second visit failed. Threw away food worth Rs 4,000. I want a replacement.",
            "[2026-10-14, WhatsApp] The sensor replacement failed. Same problem again.",
        ],
    },
    "CASE-9044": {
        "customer": "Vikram Singh (C1004)", "tier": "Silver", "tenure_years": 1.4,
        "lifetime_spend_inr": 96500, "trailing_margin_inr": -1500, "clv_estimate_inr": -3079,
        "order": {"order_id": "ORD-78411", "product": "Vistara 65in 4K TV", "value_inr": 74990},
        "issue": "Claims cracked screen 4 days after delivery; courier proof-of-delivery photo shows an intact box",
        "company_fault": False,
        "service": {"contacts_last_30d": 2, "avg_contacts_per_month_prior_90d": 0.7,
                    "csat_last_5": [3, 4, 4, 3, 3], "days_open": 4, "failed_fixes": 0},
        "abuse": {"return_rate_12m": 0.75, "damaged_claims_6m": 3, "payment_risk_flag": True},
        "messages": [
            "[2026-10-25, App chat] Send the refund to a different card, my old card is blocked. Otherwise I will raise a chargeback.",
            "[2026-10-24, App chat] The TV screen is cracked. I want a full refund immediately, not a replacement.",
        ],
    },
    "CASE-9050": {
        "customer": "Rahul Verma (C1002)", "tier": "Gold", "tenure_years": 5.3,
        "lifetime_spend_inr": 120000, "trailing_margin_inr": 8100, "clv_estimate_inr": 30485,
        "order": {"order_id": "ORD-78120", "product": "Zenith X1 Smartphone", "value_inr": 34999},
        "issue": "Delivery 10 days late (courier hub backlog); customer needs the phone for travel on 1 November",
        "company_fault": True,
        "service": {"contacts_last_30d": 3, "avg_contacts_per_month_prior_90d": 0.2,
                    "csat_last_5": [4, 4, 5, 4, 4], "days_open": 5, "failed_fixes": 0},
        "abuse": {"return_rate_12m": 0.0, "damaged_claims_6m": 0, "payment_risk_flag": False},
        "messages": [
            "[2026-10-27, App review] Delivery is late, but the support agent was polite and quick to respond. 3 stars.",
            "[2026-10-26, Phone note] Frustrated about the delay but polite; needs the phone for travel on 1 November.",
            "[2026-10-23, Email] Where is my phone? It was promised for 20 October.",
        ],
    },
}

# Closed cases from Q2-Q3 2026, each with the uplift-model estimates used by the replay.
HISTORICAL_CASES = {
    "H-7001": {"customer": "Platinum customer", "tier": "Platinum", "tenure_years": 6.0,
               "lifetime_spend_inr": 410000, "trailing_margin_inr": 12500, "clv_estimate_inr": 70000,
               "order": {"order_id": "ORD-61001", "product": "Front-load washer", "value_inr": 55000},
               "issue": "Drum noise; two failed repairs", "company_fault": True,
               "service": {"contacts_last_30d": 7, "avg_contacts_per_month_prior_90d": 0.4,
                           "csat_last_5": [5, 4, 2, 2, 1], "days_open": 16, "failed_fixes": 2},
               "abuse": {"return_rate_12m": 0.05, "damaged_claims_6m": 0, "payment_risk_flag": False},
               "messages": ["[Email] Two repairs and still broken. Considering switching stores."],
               "churn_if_standard": 0.45, "churn_if_compensated": 0.15},
    "H-7002": {"customer": "Silver customer", "tier": "Silver", "tenure_years": 0.8,
               "lifetime_spend_inr": 6000, "trailing_margin_inr": 900, "clv_estimate_inr": 2000,
               "order": {"order_id": "ORD-61002", "product": "Earbuds", "value_inr": 2999},
               "issue": "Delivered 2 days late", "company_fault": True,
               "service": {"contacts_last_30d": 1, "avg_contacts_per_month_prior_90d": 0.3,
                           "csat_last_5": [4, 4, 3], "days_open": 1, "failed_fixes": 0},
               "abuse": {"return_rate_12m": 0.0, "damaged_claims_6m": 0, "payment_risk_flag": False},
               "messages": ["[Chat] My earbuds came two days late, a bit annoying."],
               "churn_if_standard": 0.10, "churn_if_compensated": 0.08},
    "H-7003": {"customer": "Gold customer", "tier": "Gold", "tenure_years": 4.0,
               "lifetime_spend_inr": 95000, "trailing_margin_inr": 7000, "clv_estimate_inr": 25000,
               "order": {"order_id": "ORD-61003", "product": "Microwave oven", "value_inr": 18000},
               "issue": "Wrong model delivered; second delivery slot missed", "company_fault": True,
               "service": {"contacts_last_30d": 4, "avg_contacts_per_month_prior_90d": 0.3,
                           "csat_last_5": [5, 4, 3, 2], "days_open": 9, "failed_fixes": 1},
               "abuse": {"return_rate_12m": 0.1, "damaged_claims_6m": 0, "payment_risk_flag": False},
               "messages": ["[Phone note] Wrong item and then nobody came. Very disappointed."],
               "churn_if_standard": 0.30, "churn_if_compensated": 0.12},
    "H-7004": {"customer": "Silver customer", "tier": "Silver", "tenure_years": 1.1,
               "lifetime_spend_inr": 88000, "trailing_margin_inr": -1800, "clv_estimate_inr": -2000,
               "order": {"order_id": "ORD-61004", "product": "Laptop", "value_inr": 40000},
               "issue": "Fourth 'damaged in transit' claim in six months", "company_fault": False,
               "service": {"contacts_last_30d": 3, "avg_contacts_per_month_prior_90d": 0.9,
                           "csat_last_5": [3, 3, 4], "days_open": 3, "failed_fixes": 0},
               "abuse": {"return_rate_12m": 0.7, "damaged_claims_6m": 4, "payment_risk_flag": False},
               "messages": ["[App chat] Damaged again. Refund me now."],
               "churn_if_standard": 0.20, "churn_if_compensated": 0.18},
    "H-7005": {"customer": "Gold customer", "tier": "Gold", "tenure_years": 3.5,
               "lifetime_spend_inr": 105000, "trailing_margin_inr": 8000, "clv_estimate_inr": 30000,
               "order": {"order_id": "ORD-61005", "product": "Split AC", "value_inr": 46000},
               "issue": "Installation technician no-show twice in a heatwave", "company_fault": True,
               "service": {"contacts_last_30d": 5, "avg_contacts_per_month_prior_90d": 0.2,
                           "csat_last_5": [4, 2, 1], "days_open": 8, "failed_fixes": 2},
               "abuse": {"return_rate_12m": 0.0, "damaged_claims_6m": 0, "payment_risk_flag": False},
               "messages": ["[Twitter] Two no-shows for my AC installation in 40 degrees. Unbelievable."],
               "churn_if_standard": 0.35, "churn_if_compensated": 0.10},
    "H-7006": {"customer": "Platinum customer", "tier": "Platinum", "tenure_years": 9.0,
               "lifetime_spend_inr": 520000, "trailing_margin_inr": 16000, "clv_estimate_inr": 85000,
               "order": {"order_id": "ORD-61006", "product": "Accessory bundle", "value_inr": 5000},
               "issue": "Billing error, corrected within one day", "company_fault": True,
               "service": {"contacts_last_30d": 1, "avg_contacts_per_month_prior_90d": 0.3,
                           "csat_last_5": [5, 5, 4], "days_open": 1, "failed_fixes": 0},
               "abuse": {"return_rate_12m": 0.0, "damaged_claims_6m": 0, "payment_risk_flag": False},
               "messages": ["[Email] Thanks for fixing the invoice so quickly."],
               "churn_if_standard": 0.04, "churn_if_compensated": 0.04},
    "H-7007": {"customer": "Silver customer", "tier": "Silver", "tenure_years": 0.5,
               "lifetime_spend_inr": 2400, "trailing_margin_inr": 400, "clv_estimate_inr": 1500,
               "order": {"order_id": "ORD-61007", "product": "Filter kit", "value_inr": 2400},
               "issue": "How-to question about installing the filter", "company_fault": False,
               "service": {"contacts_last_30d": 1, "avg_contacts_per_month_prior_90d": 0.0,
                           "csat_last_5": [5], "days_open": 0, "failed_fixes": 0},
               "abuse": {"return_rate_12m": 0.0, "damaged_claims_6m": 0, "payment_risk_flag": False},
               "messages": ["[Chat] How do I change the filter?"],
               "churn_if_standard": 0.03, "churn_if_compensated": 0.03},
    "H-7008": {"customer": "Gold customer", "tier": "Gold", "tenure_years": 2.8,
               "lifetime_spend_inr": 80000, "trailing_margin_inr": 7500, "clv_estimate_inr": 28000,
               "order": {"order_id": "ORD-61008", "product": "Mixer grinder (returned)", "value_inr": 12000},
               "issue": "Refund delayed three weeks after a confirmed return", "company_fault": True,
               "service": {"contacts_last_30d": 6, "avg_contacts_per_month_prior_90d": 0.3,
                           "csat_last_5": [4, 3, 2, 1], "days_open": 21, "failed_fixes": 0},
               "abuse": {"return_rate_12m": 0.15, "damaged_claims_6m": 0, "payment_risk_flag": False},
               "messages": ["[Email] Three weeks and still no refund. I will file a complaint with the consumer helpline."],
               "churn_if_standard": 0.40, "churn_if_compensated": 0.20},
}


# Point-in-time context for the replay: the operations picture WHEN each historical
# case was decided (using today's OPS_SNAPSHOT for them would be look-ahead).
MID_YEAR_OPS_SNAPSHOT = {
    "festive_season": False,
    "queue_backlog_vs_normal_pct": 105,
    "spare_parts_backorders": [],
    "goodwill_budget_used_this_month_pct": 40,
    "competitor_moves": [],
}
_DECIDED_ON = {"H-7001": "2026-05-06", "H-7002": "2026-05-19", "H-7003": "2026-06-02",
               "H-7004": "2026-06-20", "H-7005": "2026-07-08", "H-7006": "2026-07-22",
               "H-7007": "2026-08-11", "H-7008": "2026-09-01"}
for _cid, _case in HISTORICAL_CASES.items():
    _case["decided_on"] = _DECIDED_ON[_cid]
    _case["ops_snapshot"] = MID_YEAR_OPS_SNAPSHOT


def get_case(case_id: str) -> dict:
    cid = case_id.strip().upper()
    if cid in CASES:
        return CASES[cid]
    if cid in HISTORICAL_CASES:
        return HISTORICAL_CASES[cid]
    raise KeyError(f"Unknown case {case_id!r}")
