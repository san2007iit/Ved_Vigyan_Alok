"""
mock_crm.py
-----------
A small, synthetic NovaMart CRM: customers, orders, support tickets and
customer interactions (messages, call notes, social posts, reviews).

In the finance edition of this lab the tools called yfinance for live
financial statements. Here every tool reads this module instead, so the
lab runs with only an OpenAI key and gives the same numbers every time.
To connect a real system, replace the functions in crm_tools.py with calls
to your CRM / order-management / ticketing APIs; the agents do not change.

All people and data are fictional. Amounts are in Indian rupees (INR).
"""

from datetime import date

AS_OF = date(2026, 10, 28)   # the "today" of the dataset

CUSTOMERS = {
    "C1001": {
        "name": "Priya Sharma", "tier": "Platinum", "city": "Pune", "region": "West",
        "member_since": "2019-03-14", "preferred_channel": "WhatsApp", "language": "English",
        "annual_spend_inr": {"2024": 61000, "2025": 88000, "2026_ytd": 74000},
        "annual_margin_inr": {"2024": 10980, "2025": 15840, "2026_ytd": 13320},
        "trailing_12m_margin_inr": 15800, "lifetime_spend_inr": 485000, "lifetime_orders": 38,
    },
    "C1002": {
        "name": "Rahul Verma", "tier": "Gold", "city": "Delhi", "region": "North",
        "member_since": "2021-07-02", "preferred_channel": "Email", "language": "Hindi",
        "annual_spend_inr": {"2024": 28000, "2025": 33500, "2026_ytd": 41000},
        "annual_margin_inr": {"2024": 5040, "2025": 6030, "2026_ytd": 7380},
        "trailing_12m_margin_inr": 8100, "lifetime_spend_inr": 120000, "lifetime_orders": 14,
    },
    "C1003": {
        "name": "Ananya Iyer", "tier": "Silver", "city": "Chennai", "region": "South",
        "member_since": "2024-11-20", "preferred_channel": "Chat", "language": "Tamil",
        "annual_spend_inr": {"2024": 18500, "2025": 1800, "2026_ytd": 2400},
        "annual_margin_inr": {"2024": 3330, "2025": 320, "2026_ytd": 430},
        "trailing_12m_margin_inr": 520, "lifetime_spend_inr": 22700, "lifetime_orders": 3,
    },
    "C1004": {
        "name": "Vikram Singh", "tier": "Silver", "city": "Jaipur", "region": "North",
        "member_since": "2025-06-01", "preferred_channel": "App", "language": "Hindi",
        "annual_spend_inr": {"2024": 0, "2025": 41000, "2026_ytd": 55500},
        "annual_margin_inr": {"2024": 0, "2025": 1200, "2026_ytd": -2400},
        "trailing_12m_margin_inr": -1500, "lifetime_spend_inr": 96500, "lifetime_orders": 7,
    },
    "C1005": {
        "name": "Meera Nair", "tier": "Gold", "city": "Kochi", "region": "South",
        "member_since": "2020-02-11", "preferred_channel": "Phone", "language": "Malayalam",
        "annual_spend_inr": {"2024": 36000, "2025": 39000, "2026_ytd": 52000},
        "annual_margin_inr": {"2024": 6480, "2025": 7020, "2026_ytd": 9360},
        "trailing_12m_margin_inr": 10100, "lifetime_spend_inr": 210000, "lifetime_orders": 22,
    },
}

ORDERS = {
    "C1001": [
        {"order_id": "ORD-77812", "date": "2026-09-12", "product": "FrostLine Refrigerator 340L", "sku": "FL-340",
         "amount_inr": 62990, "status": "DELIVERED", "delivered": "2026-09-18", "returned": False,
         "warranty_until": "2027-09-18", "repair_attempts": 2,
         "note": "Stopped cooling 2026-10-09; repairs on 10-12 and 10-19 failed; replacement under repeat-failure rule pending"},
        {"order_id": "ORD-76520", "date": "2026-05-02", "product": "BreezeMax Split AC 1.5T", "sku": "BMX-15",
         "amount_inr": 46490, "status": "DELIVERED", "delivered": "2026-05-06", "returned": False,
         "warranty_until": "2027-05-06", "repair_attempts": 0, "note": ""},
        {"order_id": "ORD-75112", "date": "2025-11-20", "product": "Vistara 55in 4K TV", "sku": "VT-55",
         "amount_inr": 52990, "status": "DELIVERED", "delivered": "2025-11-23", "returned": False,
         "warranty_until": "2026-11-23", "repair_attempts": 0, "note": ""},
        {"order_id": "ORD-71230", "date": "2024-08-03", "product": "HomeChef Mixer Grinder", "sku": "HC-750",
         "amount_inr": 4290, "status": "RETURNED", "delivered": "2024-08-05", "returned": True,
         "warranty_until": "2025-08-05", "repair_attempts": 0, "note": "Change of mind return"},
    ],
    "C1002": [
        {"order_id": "ORD-78120", "date": "2026-10-14", "product": "Zenith X1 Smartphone", "sku": "ZX1",
         "amount_inr": 34999, "status": "IN_TRANSIT", "delivered": None, "returned": False,
         "promised": "2026-10-20", "eta": "2026-10-30", "warranty_until": None, "repair_attempts": 0,
         "note": "Held at SwiftShip Delhi (Okhla) hub: festive-season backlog"},
        {"order_id": "ORD-76011", "date": "2026-03-09", "product": "Zenith Buds Pro", "sku": "ZBP",
         "amount_inr": 3499, "status": "DELIVERED", "delivered": "2026-03-11", "returned": False,
         "warranty_until": "2027-03-11", "repair_attempts": 0, "note": ""},
    ],
    "C1003": [
        {"order_id": "ORD-71999", "date": "2024-11-20", "product": "AquaPure RO Purifier", "sku": "AQP-500",
         "amount_inr": 18500, "status": "DELIVERED", "delivered": "2024-11-25", "returned": False,
         "warranty_until": "2025-11-25", "repair_attempts": 0, "note": ""},
        {"order_id": "ORD-78305", "date": "2026-10-24", "product": "AquaPure RO Filter Kit", "sku": "AQP-FK",
         "amount_inr": 2400, "status": "OUT_FOR_DELIVERY", "delivered": None, "returned": False,
         "warranty_until": None, "repair_attempts": 0, "note": ""},
    ],
    "C1004": [
        {"order_id": "ORD-78411", "date": "2026-10-18", "product": "Vistara 65in 4K TV", "sku": "VT-65",
         "amount_inr": 74990, "status": "DELIVERED", "delivered": "2026-10-20", "returned": False,
         "warranty_until": "2027-10-20", "repair_attempts": 0,
         "note": "Customer reports cracked screen on 2026-10-24 (4 days after delivery); courier POD photo shows intact box"},
        {"order_id": "ORD-77002", "date": "2026-07-11", "product": "Vistara Soundbar 5.1", "sku": "VSB-51",
         "amount_inr": 18990, "status": "RETURNED", "delivered": "2026-07-13", "returned": True,
         "warranty_until": None, "repair_attempts": 0, "note": "Return reason: damaged in transit"},
        {"order_id": "ORD-76340", "date": "2026-05-19", "product": "Zenith Watch S", "sku": "ZWS",
         "amount_inr": 12490, "status": "RETURNED", "delivered": "2026-05-21", "returned": True,
         "warranty_until": None, "repair_attempts": 0, "note": "Return reason: damaged in transit"},
        {"order_id": "ORD-75550", "date": "2026-02-02", "product": "Nimbus 14 Laptop", "sku": "NB-14",
         "amount_inr": 62990, "status": "RETURNED", "delivered": "2026-02-05", "returned": True,
         "warranty_until": None, "repair_attempts": 0, "note": "Return reason: not as described"},
    ],
    "C1005": [
        {"order_id": "ORD-78502", "date": "2026-10-19", "product": "BreezeMax Split AC 1.5T", "sku": "BMX-15",
         "amount_inr": 46490, "status": "DELIVERED", "delivered": "2026-10-22", "returned": False,
         "warranty_until": "2027-10-22", "repair_attempts": 0,
         "note": "Installed 2026-10-22; burning smell and smoke from indoor unit reported 2026-10-27"},
        {"order_id": "ORD-74410", "date": "2025-12-14", "product": "FrostLine Refrigerator 400L", "sku": "FL-400",
         "amount_inr": 71490, "status": "DELIVERED", "delivered": "2025-12-17", "returned": False,
         "warranty_until": "2026-12-17", "repair_attempts": 0, "note": ""},
    ],
}

# status: RESOLVED | OPEN | ESCALATED | REOPENED ; contacts = customer contacts on the ticket
TICKETS = {
    "C1001": [
        {"ticket_id": "TKT-41022", "opened": "2025-03-02", "channel": "Email", "category": "Delivery query",
         "first_response_hours": 3.0, "resolution_hours": 20.0, "status": "RESOLVED", "contacts": 1,
         "csat": 5, "escalated": False},
        {"ticket_id": "TKT-45310", "opened": "2025-12-01", "channel": "Chat", "category": "Installation slot",
         "first_response_hours": 0.2, "resolution_hours": 4.0, "status": "RESOLVED", "contacts": 1,
         "csat": 5, "escalated": False},
        {"ticket_id": "TKT-50211", "opened": "2026-10-09", "channel": "WhatsApp", "category": "Product defect",
         "first_response_hours": 2.5, "resolution_hours": None, "status": "REOPENED", "contacts": 3,
         "csat": None, "escalated": False},
        {"ticket_id": "TKT-50388", "opened": "2026-10-13", "channel": "Phone", "category": "Repair failed",
         "first_response_hours": 6.0, "resolution_hours": None, "status": "ESCALATED", "contacts": 2,
         "csat": 2, "escalated": True},
        {"ticket_id": "TKT-50642", "opened": "2026-10-20", "channel": "Email", "category": "Replacement request",
         "first_response_hours": 20.0, "resolution_hours": None, "status": "ESCALATED", "contacts": 2,
         "csat": 1, "escalated": True},
        {"ticket_id": "TKT-50701", "opened": "2026-10-26", "channel": "WhatsApp", "category": "Complaint",
         "first_response_hours": 1.5, "resolution_hours": None, "status": "OPEN", "contacts": 1,
         "csat": None, "escalated": True},
    ],
    "C1002": [
        {"ticket_id": "TKT-38811", "opened": "2025-04-18", "channel": "Email", "category": "Invoice copy",
         "first_response_hours": 5.0, "resolution_hours": 9.0, "status": "RESOLVED", "contacts": 1,
         "csat": 4, "escalated": False},
        {"ticket_id": "TKT-50890", "opened": "2026-10-23", "channel": "Email", "category": "Where is my order",
         "first_response_hours": 9.0, "resolution_hours": None, "status": "OPEN", "contacts": 2,
         "csat": None, "escalated": False},
        {"ticket_id": "TKT-50951", "opened": "2026-10-26", "channel": "Phone", "category": "Delivery delay",
         "first_response_hours": 0.5, "resolution_hours": None, "status": "OPEN", "contacts": 1,
         "csat": None, "escalated": False},
    ],
    "C1003": [
        {"ticket_id": "TKT-47720", "opened": "2026-08-12", "channel": "Chat", "category": "How-to: filter change",
         "first_response_hours": 0.2, "resolution_hours": 0.5, "status": "RESOLVED", "contacts": 1,
         "csat": 5, "escalated": False},
    ],
    "C1004": [
        {"ticket_id": "TKT-46001", "opened": "2026-02-07", "channel": "App", "category": "Return request",
         "first_response_hours": 4.0, "resolution_hours": 52.0, "status": "RESOLVED", "contacts": 3,
         "csat": 3, "escalated": False},
        {"ticket_id": "TKT-47502", "opened": "2026-05-22", "channel": "App", "category": "Damaged in transit",
         "first_response_hours": 3.0, "resolution_hours": 30.0, "status": "RESOLVED", "contacts": 2,
         "csat": 4, "escalated": False},
        {"ticket_id": "TKT-48810", "opened": "2026-07-14", "channel": "App", "category": "Damaged in transit",
         "first_response_hours": 6.0, "resolution_hours": 41.0, "status": "RESOLVED", "contacts": 2,
         "csat": 4, "escalated": False},
        {"ticket_id": "TKT-50820", "opened": "2026-10-24", "channel": "App", "category": "Damaged on arrival",
         "first_response_hours": 2.0, "resolution_hours": None, "status": "OPEN", "contacts": 2,
         "csat": None, "escalated": False},
    ],
    "C1005": [
        {"ticket_id": "TKT-39920", "opened": "2025-01-09", "channel": "Phone", "category": "Installation slot",
         "first_response_hours": 0.3, "resolution_hours": 2.0, "status": "RESOLVED", "contacts": 1,
         "csat": 5, "escalated": False},
        {"ticket_id": "TKT-51002", "opened": "2026-10-27", "channel": "Phone", "category": "Product safety",
         "first_response_hours": 0.4, "resolution_hours": None, "status": "ESCALATED", "contacts": 1,
         "csat": None, "escalated": True},
    ],
}

# Voice of the customer: what the customer said, where, and when.
INTERACTIONS = {
    "C1001": [
        {"date": "2026-10-09", "source": "WhatsApp", "text": "My new FrostLine fridge stopped cooling. Please send someone today."},
        {"date": "2026-10-14", "source": "WhatsApp", "text": "The sensor replacement failed. Same problem again. Very disappointed."},
        {"date": "2026-10-20", "source": "Email", "text": "Second technician visit failed. I threw away food worth Rs 4,000. I want a replacement, not another repair."},
        {"date": "2026-10-25", "source": "Twitter", "text": "@NovaMartCare 3 weeks without a working fridge after paying Rs 63k. Two failed repairs. Is this how you treat a 7-year Platinum customer? #NovaMartFail"},
        {"date": "2026-10-26", "source": "Phone call note", "text": "Customer said this is unacceptable; will move to another store and send a consumer court notice if not replaced by Thursday."},
    ],
    "C1002": [
        {"date": "2026-10-23", "source": "Email", "text": "Where is my phone? It was promised for 20 October. Tracking has not moved for days."},
        {"date": "2026-10-26", "source": "Phone call note", "text": "Customer frustrated about the delay but polite; needs the phone for travel on 1 November."},
        {"date": "2026-10-27", "source": "App review", "text": "Delivery is late, but the support agent was polite and quick to respond. 3 stars."},
    ],
    "C1003": [
        {"date": "2026-08-12", "source": "Chat", "text": "Thanks, very helpful! The filter video made it easy."},
        {"date": "2026-10-24", "source": "App review", "text": "Great purifier, quick delivery of the filter kit. Happy customer."},
    ],
    "C1004": [
        {"date": "2026-10-24", "source": "App chat", "text": "The TV screen is cracked. I want a full refund immediately, not a replacement."},
        {"date": "2026-10-25", "source": "App chat", "text": "Send the refund to a different card, my old card is blocked. Otherwise I will raise a chargeback with my bank."},
    ],
    "C1005": [
        {"date": "2026-10-27", "source": "Phone call note", "text": "Burning smell and some smoke from the AC indoor unit last night. Switched it off. Children at home, very worried."},
        {"date": "2026-10-27", "source": "Email", "text": "Please send someone urgently and confirm the unit is safe. I have always trusted NovaMart."},
    ],
}

# Segment (loyalty-tier) benchmarks: the "peer group" of a customer.
SEGMENT_BENCHMARKS = {
    "Platinum": {"avg_annual_spend_inr": 78000, "avg_order_value_inr": 11500, "avg_annual_margin_inr": 14000,
                 "contacts_per_order": 0.35, "return_rate": 0.06, "baseline_retention": 0.92},
    "Gold":     {"avg_annual_spend_inr": 32000, "avg_order_value_inr": 7800, "avg_annual_margin_inr": 5760,
                 "contacts_per_order": 0.45, "return_rate": 0.08, "baseline_retention": 0.85},
    "Silver":   {"avg_annual_spend_inr": 9500, "avg_order_value_inr": 4200, "avg_annual_margin_inr": 1710,
                 "contacts_per_order": 0.55, "return_rate": 0.10, "baseline_retention": 0.72},
}


def lookup(table: dict, customer_id: str):
    """Return a customer's record from one of the tables above, or raise KeyError."""
    cid = (customer_id or "").strip().upper()
    if cid not in table:
        raise KeyError(f"Unknown customer ID {cid!r}")
    return table[cid]
