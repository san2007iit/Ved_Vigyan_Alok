"""
common.py
=========
Shared tooling for the Chapter 3 framework bake-off (customer-care edition).

Every implementation runs the same task against the same deterministic
mock data, so differences in the outputs reflect the frameworks, not
the inputs. Swapping in a live CRM / ticketing source (Zendesk, Freshdesk,
Salesforce Service Cloud) is a one-line change in MOCK_SUPPORT_DATA /
get_support_data_tool_func.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Deterministic mock data (Q3 2026). Live alternative: a ticketing-system API
# call such as GET /api/v2/tickets?product=ZX1&period=2026-Q3.
MOCK_SUPPORT_DATA = {
    "ZX1": {"product": "Zenith X1 Smartphone", "orders": 18400, "tickets": 2760},
    "FL-340": {"product": "FrostLine Refrigerator 340L", "orders": 3150, "tickets": 1134},
}

# One small model for every OpenAI-backed implementation so the
# comparison stays fair. Update to the current small tier before running.
OPENAI_MODEL = os.getenv("BAKEOFF_OPENAI_MODEL", "gpt-5-mini")

system_message = (
    "You are a precise customer-experience analyst. Use the provided tools to "
    "fetch support data and compute contact rates - never invent numbers. "
    "Finish with a short, professional memo for the Head of Customer Care."
)

input_message = (
    "Compare the Zenith X1 smartphone (ZX1) and the FrostLine refrigerator "
    "(FL-340) on contact rate (support tickets per 100 orders) and summarize "
    "in a memo."
)


@dataclass
class SupportData:
    product_line: str
    product: str
    orders: int
    tickets: int


def check_api_key(name: str = "OPENAI_API_KEY") -> None:
    if not os.getenv(name):
        raise SystemExit(
            f"{name} is not set. Export it (or add it to .env) first.")


def get_support_data_tool_func(product_line: str) -> SupportData:
    """Return the quarter's orders and support tickets for a product line (mock data)."""
    row = MOCK_SUPPORT_DATA.get(product_line.strip().upper())
    if row is None:
        raise ValueError(f"No data for product line {product_line!r}")
    return SupportData(product_line=product_line.strip().upper(), product=row["product"],
                       orders=row["orders"], tickets=row["tickets"])


def compute_contact_rate_tool_func(tickets: int, orders: int) -> float:
    """Compute the contact rate: support tickets per 100 orders."""
    if orders == 0:
        raise ValueError("Orders is zero; contact rate undefined")
    return round(tickets / orders * 100, 2)


def describe(d: SupportData) -> str:
    return f"{d.product_line} ({d.product}): orders={d.orders} tickets={d.tickets}"


def persist_framework_output(framework: str, output: str,
                             token_usage: str = "") -> Path:
    """Write a framework's run output to outputs/<framework>.txt."""
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{framework}.txt"
    header = f"# {framework} - bake-off output\n"
    if token_usage:
        header += f"# token usage: {token_usage}\n"
    path.write_text(header + "\n" + output.strip() + "\n", encoding="utf-8")
    print(f"[saved] {path}")
    return path
