# Self-Consistency - Remedy Decision Lab (customer-care edition)

An implementation of the **Self-Consistency** reasoning paradigm (Wang et al., 2022)
applied to an after-sales decision: for an ambiguous AC complaint, recommend
**REPAIR / REPLACE / REFUND** by sampling several independent chain-of-thought
reasoning paths and taking the plurality vote.

*Finance edition*: BUY / HOLD / SELL for a stock.

Why this case is ambiguous on purpose: a repair is available in 3 days with ~80%
success, a replacement takes 9 days and is not yet mandatory (only one failed
repair), a refund is allowed only if a defect is confirmed, and the customer is a
Gold member working from home in a heatwave. Reasonable specialists disagree:
exactly the situation where one answer is risky and a vote helps.

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # then add ANTHROPIC_API_KEY
python self-consistency.py
```

The script prints each sampled reasoning path with its parsed recommendation, then
the vote breakdown, the agreement ratio and the majority recommendation. If fewer
than 60% of the paths agree, it recommends routing the case to a human.

## Configuration

- `MODEL` - default `claude-opus-5-5` (override with `CC_ANTHROPIC_MODEL`).
- `N_PATHS` - number of independent reasoning paths (default `5`).

The finance lab set `TEMPERATURE = 0.8` to diversify the paths. Claude models from
Opus 4.7 onwards no longer accept a temperature setting, so this edition relies on
default sampling; the paths still differ, but less. The **agreement ratio** is the
business signal: low agreement = an uncertain decision that deserves a human.
