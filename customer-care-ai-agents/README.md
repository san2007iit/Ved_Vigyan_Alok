# Building AI Agents for Customer Care: course code for an MBA class

A customer-care edition of the code that accompanies *Building AI Agents for Finance*.
Every finance lab (stock analysis, portfolio memory, investment committee, KYC, insurance
claims...) is re-built on a single fictional company, **NovaMart**, an Indian electronics and
appliance retailer, so a class can follow one business through all twelve chapters.

**Frameworks, notebook-versus-script style and chapter structure match the original.**
The teaching explanation of how each chapter works is in [`TEACHING_GUIDE.md`](TEACHING_GUIDE.md).

## The NovaMart cast

| ID | Customer | Situation |
|---|---|---|
| C1001 | Priya Sharma, Platinum, Pune | FrostLine refrigerator (Rs 62,990), two failed repairs, a Twitter post, a consumer-court threat |
| C1002 | Rahul Verma, Gold, Delhi | Phone delivery 8+ days late, polite but frustrated |
| C1003 | Ananya Iyer, Silver, Chennai | Happy, low-value; a water purifier and a filter kit |
| C1004 | Vikram Singh, Silver, Jaipur | Repeated "damaged" claims, refund to a different card, negative margin |
| C1005 | Meera Nair, Gold, Kochi | Smoke from a new AC: a safety escalation |

Policies used throughout: **KB-101** 10-day replacement window; **KB-102** repeat failure means
replacement; **KB-103** goodwill authority limits; **KB-104** delay compensation; **KB-105** safety
escalation. All people, orders and numbers are synthetic. "Today" in the data is 28 Oct 2026.

## Finance to customer care, at a glance

| Ch | Finance edition | Customer-care edition |
|---|---|---|
| 1 | Earnings-call summary, knowledge cutoff, price/news agents | Support-call summary, knowledge cutoff, order-status/service-alert agents |
| 2 | Fundamental ratios, portfolio memory, ReAct stock price, news evaluator-optimizer, investment sequence | Customer KPIs, conversation memory, ReAct shipment tracking, complaint-reply evaluator-optimizer, retention-decision sequence |
| 3 | Nine-framework bake-off on an EV/EBITDA task | Nine frameworks on a **contact-rate** task, plus a decision matrix |
| 4 | Investment-research orchestrator (DCF) | **Case-resolution** orchestrator (customer lifetime value) |
| 5 | Deep-search financial research | Deep-search **CX research** |
| 6 | Self-refine, self-consistency, ReAct, Tree of Thoughts, LATS | Same patterns on apology letters, remedy choice, peer benchmarks, recovery plans, staffing |
| 7 | Multi-agent company health | Multi-agent **B2B account health** |
| 8 | Investment committee, bull/bear debate, backtest | **Escalation committee**, advocate/controller debate, case replay |
| 9 | Insurance claims, fraud, underwriting | **Warranty claims**, return-abuse, Care+ pricing |
| 10 | Agentic RAG over a 10-K | Agentic RAG over CX reports and a service policy |
| 11 | KYC evaluation harness | **Complaint-resolution** evaluation harness |
| 12 | Observability and guardrails | Same, on the case-resolution workflow |

## Chapter index

| Folder | Contents |
|---|---|
| `Chapter 1` | 3 notebooks |
| `Chapter 2` | 6 notebooks |
| `Chapter 3` | 9 notebooks, a script per framework in `bakeoff/`, decision-matrix `templates/` |
| `Chapter 4` | `chapter_4_lab_case_resolution_agent/` (multi-agent CLI app) |
| `Chapter 5` | 2 notebooks and `src/` deep-search pipeline |
| `Chapter 6` | 5 lab folders (scripts) |
| `Chapter 7` | 5 notebooks |
| `Chapter 8` | LangGraph committee, debate, replay (scripts) |
| `Chapter 9` | 3 notebooks and `src/` |
| `Chapter 10` | `agentic_rag_lab.py` and 3 documents |
| `Chapter 11` | Evaluation harness package |
| `Chapter 12` | Workflow, guardrails, observability, 4 offline snippets |

## Setup

* **Colab**: open a notebook, add the keys it needs under the key icon, upload the chapter's
  `common.py` if it has one (or set `CC_RAW_BASE` to the raw URL of your copy), Run all.
* **Local**: `python -m venv .venv`, activate, `pip install -r requirements.txt` inside the chapter
  folder, copy `.env.sample` to `.env`. Each chapter folder is self-contained.

Use a separate virtual environment per chapter: the frameworks pin conflicting versions of
protobuf, httpx and pydantic.

## What was tested, and how

Every notebook and script was executed end to end against a **local stand-in for the OpenAI and
Anthropic APIs** (no real keys, no network), which exercises the full code path: imports, tool
calls, handoffs, streaming, parsing, file output. That checks the mechanics, **not the quality of
real model answers**: run each lab once with real keys before teaching it. Not executed end to
end: the Claude Agent SDK labs (Ch 3 `claude_agent_sdk_agent.py`, Ch 6 ReAct and LATS), which need
the Claude Code CLI; their tool handlers and search logic were tested separately.

Real-library issues found while converting and fixed in the code:

* `Runner.run_sync()` fails inside a notebook's running event loop, so notebooks use `await Runner.run(...)`.
* CrewAI in a notebook needs `await crew.kickoff_async()`.
* AutoGen's classic API is now the `ag2` package (`ag2[openai]<1.0`), not `pyautogen`.
* `llama-index-llms-anthropic` 0.12.x breaks with `anthropic` 2.x, so Chapter 10 pins `anthropic<2`.

## Disclaimer

Educational use only. NovaMart, its customers and all figures are fictional. Real deployments
need human approval of commitments, legal review of denials, and fairness testing (Chapters 11-12).
