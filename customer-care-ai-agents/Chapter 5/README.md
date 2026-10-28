# Chapter 5 - Deep search: analyst-grade customer-experience research

*Finance edition*: a Deep Search Agent researching NVIDIA vs AMD vs Intel.
*Customer-care edition*: the same architecture investigates **why FrostLine
refrigerator CSAT fell in Q3 2026**, compared with two healthy product lines.

```
Plan -> Execute (parallel waves) -> Validate (deterministic) -> (Replan?) -> Synthesize
```

Two ways to work through this chapter's code:

- **Lab notebooks** (start here):
  - `chapter_5_lab_1_deep_search_planner.ipynb` - the planner in isolation:
    decompose a research question into typed, dependency-ordered sub-tasks.
  - `chapter_5_lab_2_deep_search_full_pipeline.ipynb` - the full pipeline:
    planner, researcher, validator and synthesizer working end to end (stub data tools).
- **`src/`** - the complete project: `planner.py`, `researcher.py`, `validator.py`,
  `synthesizer.py`, `deep_search_agent.py`, `tools/`, shared `common.py`.

```bash
cd src
pip install -r requirements.txt
export OPENAI_API_KEY=...            # or put it in a .env file
python deep_search_agent.py          # writes output/report_*.md and .json
python deep_search_agent.py -q "Why are AC installation complaints rising?" -p BMX-15,AQP-500
```

## Data sources (finance -> customer care)

| Source | Customer-care tool | Finance-edition tool |
|---|---|---|
| `support_analytics` | `tools/support_analytics.py`: orders, tickets, contact rate, CSAT and its change, FCR, repeat contacts, resolution time, returns per product line | Financial Datasets API (revenue, EPS, margins, P/E) |
| `cx_documents` | `tools/cx_documents.py`: engineering known-issue bulletins, field-service reports, QA call audits | SEC EDGAR risk factors |
| `web_search` | `tools/web_search.py`: social media, review sites, forums, competitor moves (live Tavily if `TAVILY_API_KEY` is set, otherwise clearly-labelled synthetic results) | Tavily financial news |

Only `OPENAI_API_KEY` is required. All support data is synthetic (fictional NovaMart).

## The deterministic validator

Before anything reaches a memo, `validator.py` checks with plain Python that every
product line is covered and that the numbers are possible: CSAT within 1-5, rates
within 0-100%, the **contact rate equals tickets / orders x 100**, and **first-contact
resolution + repeat-contact rate do not exceed 100%** (a ticket cannot be both). Big
quarter-on-quarter CSAT moves are flagged as warnings.
