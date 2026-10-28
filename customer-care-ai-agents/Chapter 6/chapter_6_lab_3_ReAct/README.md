# ReAct Lab (customer-care edition)

A ReAct agent built on the **Claude Agent SDK** that decides whether a customer's
contact pattern is abnormal for their loyalty tier and whether a late delivery
justifies **proactive compensation**, using three tools over a mock CRM.

*Finance edition*: "Is Apple overvalued relative to the Technology sector?" with
yfinance tools (valuation ratios, sector median P/E, balance sheet).

| Finance tool | Customer-care tool |
|---|---|
| `get_valuation_ratios(ticker)` | `get_customer_metrics(customer_id)`: tier, contacts per order, CSAT, open tickets |
| `get_sector_median_pe(sector)` | `get_segment_median(tier)`: median contacts per order and CSAT across the tier |
| `get_balance_sheet(ticker)` | `get_order_history(customer_id)`: orders, days late, delay-compensation policy KB-104 |

The agent alternates **Thought -> Action -> Observation** steps, printed to the console,
and ends with a recommendation.

## Prerequisites

- Python 3.10+
- Node.js 18+ (the Claude Agent SDK launches the Claude Code CLI under the hood)
- An Anthropic API key

```bash
npm install -g @anthropic-ai/claude-code
export ANTHROPIC_API_KEY="sk-ant-..."     # PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python react.py
```

You should see a trace like:

```
[THOUGHT] I need C1002's metrics first...
[ACTION] mcp__care__get_customer_metrics({'customer_id': 'C1002'})
[OBSERVATION] {"tier": "Gold", "contacts_per_order_90d": 3.0, ...}
[THOUGHT] Now the Gold-tier median to benchmark...
[ACTION] mcp__care__get_segment_median({'tier': 'Gold'})
...
```

## Customise

- Change the question at the bottom of `react.py` (try C1001 against the Platinum tier).
- Add customers to `CUSTOMER_METRICS` to widen the tier samples.
- Change the model with `CC_ANTHROPIC_MODEL`.
