"""ReAct agent on the Claude Agent SDK - customer-care edition.

The agent alternates Thought -> Action -> Observation steps (printed to the
console) to answer: "Is this customer's contact pattern abnormal for their
tier, and does a late delivery justify proactive compensation?"

Finance edition: "Is Apple overvalued relative to the Technology sector?",
with yfinance tools for valuation ratios, sector medians and the balance sheet.
Here the three tools read a mock CRM: customer metrics, tier medians, and the
order / delivery history with the applicable compensation policy.
"""
import asyncio
import json
import os
import statistics

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("CC_ANTHROPIC_MODEL", "claude-opus-5-5")

# Mock CRM: 90-day service metrics for a sample of customers per tier.
CUSTOMER_METRICS = {
    # Gold tier
    "C1002": {"tier": "Gold", "contacts_90d": 3, "orders_90d": 1, "avg_csat_90d": None, "open_tickets": 2},
    "C2101": {"tier": "Gold", "contacts_90d": 1, "orders_90d": 2, "avg_csat_90d": 4.5, "open_tickets": 0},
    "C2102": {"tier": "Gold", "contacts_90d": 0, "orders_90d": 1, "avg_csat_90d": None, "open_tickets": 0},
    "C2103": {"tier": "Gold", "contacts_90d": 2, "orders_90d": 3, "avg_csat_90d": 4.0, "open_tickets": 0},
    "C2104": {"tier": "Gold", "contacts_90d": 1, "orders_90d": 1, "avg_csat_90d": 5.0, "open_tickets": 1},
    "C2105": {"tier": "Gold", "contacts_90d": 4, "orders_90d": 2, "avg_csat_90d": 3.0, "open_tickets": 1},
    "C1005": {"tier": "Gold", "contacts_90d": 1, "orders_90d": 1, "avg_csat_90d": None, "open_tickets": 1},
    # Platinum tier
    "C1001": {"tier": "Platinum", "contacts_90d": 9, "orders_90d": 1, "avg_csat_90d": 1.5, "open_tickets": 4},
    "C3101": {"tier": "Platinum", "contacts_90d": 1, "orders_90d": 3, "avg_csat_90d": 4.8, "open_tickets": 0},
    "C3102": {"tier": "Platinum", "contacts_90d": 2, "orders_90d": 4, "avg_csat_90d": 4.6, "open_tickets": 0},
    "C3103": {"tier": "Platinum", "contacts_90d": 0, "orders_90d": 2, "avg_csat_90d": None, "open_tickets": 0},
}

ORDER_HISTORY = {
    "C1002": [
        {"order_id": "ORD-78120", "product": "Zenith X1 Smartphone", "amount_inr": 34999,
         "promised": "2026-10-20", "status": "IN_TRANSIT", "eta": "2026-10-30", "days_late_at_eta": 10},
        {"order_id": "ORD-76011", "product": "Zenith Buds Pro", "amount_inr": 3499,
         "promised": "2026-03-11", "status": "DELIVERED", "eta": None, "days_late_at_eta": 0},
    ],
    "C1001": [
        {"order_id": "ORD-77812", "product": "FrostLine Refrigerator 340L", "amount_inr": 62990,
         "promised": "2026-09-18", "status": "DELIVERED", "eta": None, "days_late_at_eta": 0},
    ],
}

DELAY_POLICY = ("KB-104: delivered 3+ days late -> Rs 250 voucher; 5+ days late -> delivery-fee refund "
                "plus Rs 500 voucher; Platinum members receive double the voucher value.")


def _contacts_per_order(m: dict) -> float | None:
    return round(m["contacts_90d"] / m["orders_90d"], 2) if m["orders_90d"] else None


def _text(data: dict) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(data)}]}


@tool("get_customer_metrics",
      "Retrieve a customer's 90-day service metrics: tier, contacts per order, average CSAT, open tickets.",
      {"customer_id": str})
async def get_customer_metrics(args):
    m = CUSTOMER_METRICS.get(args["customer_id"].strip().upper())
    if m is None:
        return _text({"error": f"Unknown customer {args['customer_id']}"})
    return _text({**m, "contacts_per_order_90d": _contacts_per_order(m)})


@tool("get_segment_median",
      "Retrieve the median contacts per order and median CSAT across customers of a loyalty tier.",
      {"tier": str})
async def get_segment_median(args):
    peers = [m for m in CUSTOMER_METRICS.values() if m["tier"].lower() == args["tier"].strip().lower()]
    cpo = [_contacts_per_order(m) for m in peers if _contacts_per_order(m) is not None]
    csat = [m["avg_csat_90d"] for m in peers if m["avg_csat_90d"] is not None]
    return _text({
        "median_contacts_per_order": round(statistics.median(cpo), 2) if cpo else None,
        "median_csat": round(statistics.median(csat), 2) if csat else None,
        "sample_size": len(peers),
    })


@tool("get_order_history",
      "Retrieve a customer's orders with delivery status, days late, and the delivery-delay compensation policy.",
      {"customer_id": str})
async def get_order_history(args):
    orders = ORDER_HISTORY.get(args["customer_id"].strip().upper())
    if orders is None:
        return _text({"error": f"No orders for {args['customer_id']}"})
    return _text({"orders": orders, "applicable_policy": DELAY_POLICY})


SYSTEM_PROMPT = """You are a NovaMart customer-experience analyst following the ReAct pattern.
For every step, first emit a 'Thought:' line explaining your plan or interpretation,
then call the appropriate tool. After each observation, briefly reason about what it
means before the next step. Conclude with a clear, well-grounded recommendation that
cites the numbers and the policy you used."""


async def run_react_agent(question: str) -> str:
    server = create_sdk_mcp_server(
        name="care",
        version="1.0.0",
        tools=[get_customer_metrics, get_segment_median, get_order_history],
    )
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"care": server},
        allowed_tools=[
            "mcp__care__get_customer_metrics",
            "mcp__care__get_segment_median",
            "mcp__care__get_order_history",
        ],
    )

    final = ""
    async with ClaudeSDKClient(options=options) as client:
        await client.query(question)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(f"\n[THOUGHT] {block.text.strip()}")
                        final = block.text
                    elif isinstance(block, ToolUseBlock):
                        print(f"\n[ACTION] {block.name}({block.input})")
            elif isinstance(msg, UserMessage):
                content = msg.content if isinstance(msg.content, list) else []
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        print(f"[OBSERVATION] {block.content}")
    return final


if __name__ == "__main__":
    question = (
        "Is customer C1002's contact pattern abnormal compared with other Gold customers, "
        "and does the late delivery of their order justify proactive compensation? "
        "What should the care team do today?"
    )
    answer = asyncio.run(run_react_agent(question))
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
