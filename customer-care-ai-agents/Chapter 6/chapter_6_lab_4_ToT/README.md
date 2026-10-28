# Tree of Thoughts - Service-Recovery Strategy Lab (customer-care edition)

A minimal implementation of the **Tree of Thoughts (ToT)** reasoning paradigm
(Yao et al., 2023) applied to a service-recovery decision: how to win back a
high-value Platinum customer whose refrigerator failed twice, who has posted
publicly and threatened legal action, while a competitor advertises a 7-day
replacement guarantee.

*Finance edition*: choosing how to hedge a concentrated stock position into earnings.

The script explores options as a tree, uses the LLM as an evaluator, and keeps the
top branches with **beam search**:

- **Level 1**: propose 4 distinct recovery strategies (e.g. express replacement,
  free upgrade, refund plus goodwill, loaner plus repair); score each 1-10 on
  retention / public-risk impact, cost, and policy and precedent risk; keep the top `beam_width`.
- **Level 2**: for each retained strategy, propose 3 implementations (credit amount,
  timeline, who calls, exact promise); score each; keep the best.

Every LLM answer uses Claude **structured outputs** (`output_config.format` with a
JSON schema), so the tree code never parses free text.

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # then add ANTHROPIC_API_KEY
python tot.py
```

The script prints each level's candidates, their scores, the retained beam, and a
one-line summary of each retained strategy with its best implementation.

## Configuration

- `MODEL` - default `claude-opus-5-5` (override with `CC_ANTHROPIC_MODEL`).
- `beam_width` and the `n` arguments of `generate_strategies` / `generate_implementations`
  (the branching factor) are set inside `best_recovery`.

## Class exercise

Run it twice. Does the winning strategy change? Then ask: which of the scoring
criteria would *your* company weight most, and who should own those weights?
