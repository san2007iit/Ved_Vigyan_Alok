# LATS: Language Agent Tree Search (customer-care edition)

A minimal reference implementation of **Language Agent Tree Search**
([Zhou et al., 2023](https://arxiv.org/abs/2310.04406)) on the Claude Agent SDK.
LATS combines LLM-driven agents with Monte Carlo Tree Search.

The task is deliberately small so the focus stays on the search algorithm: a contact
centre must cut the **average customer wait from 40 to 30 minutes** by adding capacity
(overtime, cross-skilled agents, callbacks) to some of its five queues, without
overspending. *Finance edition*: trimming a portfolio's tech exposure from 40% to 30%
with minimal slippage.

## How it works

```
            root (40.0 min)
           /              \
      c1 (33.6 min)    c2 (35.2 min)      <- Iteration 1: expand root
      /         \
 c1/c1 (27.2)  c1/c2 (28.8)               <- Iteration 2: UCT picks c1, target reached
```

Each iteration runs four steps:

1. **Selection** - descend from the root via UCT (Upper Confidence Bound for Trees),
   balancing high-scoring branches (exploitation) against rarely visited ones (exploration).
2. **Expansion** - the LM proposes *k* distinct staffing actions in one call; each is
   applied to a copy of the parent state by a pure environment function. The environment
   rejects impossible actions (unknown queue, more than 25% extra capacity).
3. **Evaluation** - the LM scores each new state 1-10 with the `staffing-evaluator`
   skill (normalised to [0, 1] for UCT).
4. **Backpropagation** - scores propagate from new leaves back to the root.

On failure, a **reflection** step (`staffing-reflector` skill) writes a
`When <situation>, do <action>` lesson that is prepended to the next iteration's prompt
as long-term memory.

Environment arithmetic (in `lats_helpers.apply_action`): each action costs
`5 + 2 x queues + 1.5 x pct` thousand rupees per day and lowers the average wait by
`0.4 x pct x queues` minutes. Replace it with your WFM system's queue model (Erlang C)
to make it real.

## Project structure

```
chapter_6_lab_5_lats/
├── lats_search.py       # LATS algorithm: Node, UCT, select, expand, backprop, driver
├── lats_helpers.py      # environment, LM-call wrappers, parsers, output utilities
├── requirements.txt     # claude-agent-sdk, anyio, python-dotenv
└── .claude/skills/
    ├── staffing-evaluator/   # scoring rubric for the evaluation step
    └── staffing-reflector/   # reflection template for the failure path
```

## Setup and run

Prerequisites: Python 3.10+, Node.js (the Claude Agent SDK runs the Claude Code CLI),
an Anthropic API key.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
npm install -g @anthropic-ai/claude-code
cp .env.example .env             # then add ANTHROPIC_API_KEY
python lats_search.py
```

Tune the search at the bottom of `lats_search.py`: `anyio.run(lats_search, n_iterations, k_expand, max_depth)`.

## Discussion

- The evaluator's rubric is a *business policy* (cost ceilings, burnout risk). Who should own it?
- What is the analogue of "slippage" in a contact centre, and why is spreading capacity
  across queues usually safer than concentrating it?
