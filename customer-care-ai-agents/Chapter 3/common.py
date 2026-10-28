"""
common.py - Shared task setup for the Chapter 3 framework labs (customer-care edition).

Every framework lab in this chapter solves the same reference task:
compare two NovaMart product lines on CONTACT RATE (support tickets per
100 orders), then write a short memo for the Head of Customer Care.
This file centralises the typed tool inputs, the support dataset, the two
tools, and the system / user prompts, so each lab focuses on the
framework-specific code, not on plumbing.

Finance edition of the same task: compare Apple and JPMorgan on P/E
(price / earnings per share). Contact rate (tickets / orders) is the
customer-care analogue: a simple ratio that normalises one number by another.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ProductInput(BaseModel):
    product_line: str = Field(..., description="Product-line code such as ZX1 or FL-340")


class ContactRateInput(BaseModel):
    tickets: int = Field(..., description="Support tickets raised in the quarter")
    orders: int = Field(..., description="Orders delivered in the quarter")


# Reproducible mini-dataset (Q3 2026) so the labs run without a CRM connection.
support_data: dict[str, dict[str, object]] = {
    "ZX1":    {"product": "Zenith X1 Smartphone",        "orders": 18400, "tickets": 2760},
    "FL-340": {"product": "FrostLine Refrigerator 340L", "orders": 3150,  "tickets": 1134},
}


def get_support_data(product_line: str) -> ContactRateInput:
    """Get the quarter's support tickets and delivered orders for a product line."""
    code = product_line.strip().upper()
    if code not in support_data:
        raise ValueError(f"Product line {code!r} not found")
    row = support_data[code]
    return ContactRateInput(tickets=row["tickets"], orders=row["orders"])


def compute_contact_rate(tickets: int, orders: int) -> float:
    """Compute the contact rate: support tickets per 100 orders."""
    return round(tickets / orders * 100, 2)


system_message = "You are a helpful customer-experience analyst."
input_message = (
    "Compare the Zenith X1 smartphone (ZX1) and the FrostLine refrigerator (FL-340) "
    "on contact rate (support tickets per 100 orders) and summarize in a memo."
)
