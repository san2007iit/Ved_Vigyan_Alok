# Self-Refine - Escalation Letter Lab (customer-care edition)

An implementation of the **Self-Refine** reasoning paradigm (Madaan et al., 2023)
applied to customer care: drafting the **resolution letter** for an escalated
complaint (Priya Sharma's refrigerator, two failed repairs, consumer-court threat),
then iteratively critiquing and refining it with a single LLM acting as generator,
reviewer and refiner.

*Finance edition*: the same loop refined an equity investment thesis.

## Requirements

- Python 3.10+
- An Anthropic API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # then add ANTHROPIC_API_KEY
```

## Run

```bash
python self-refine.py
```

The script prints the initial letter, then for each iteration the reviewer's
feedback and the refined letter, and finally the converged letter.

## Configuration

Constants near the top of `self-refine.py`:

- `MODEL` - Claude model id (default `claude-opus-5-5`; override with `CC_ANTHROPIC_MODEL`).
- `MAX_ITERATIONS` - maximum feedback/refine rounds (default `3`).

The loop stops early when the reviewer returns a line starting with `NO_FURTHER_FEEDBACK`.

## What to look for

- Iteration 1 feedback typically catches a missing date, an unapproved promise, or a
  defensive phrase ("as per policy...").
- Every fact must come from the case block; anything else is tagged `[unverified]`.
- The prompt grows with every iteration (the history is resent), so cost grows too:
  the `MAX_ITERATIONS` cap is a business control, not just a technical one.
