# Case Resolution Agent - Lab (customer-care edition of Chapter 4)

A multi-agent customer-case analysis pipeline built with the **OpenAI Agents SDK**.
Five specialist agents (customer profile, voice of the customer, churn risk,
customer value, resolution writer) collaborate under an Orchestrator to produce a
**Case Resolution Brief** for an escalated NovaMart customer: a service tier, a
resolution plan within policy, and a draft message to the customer.

*Finance edition*: the same architecture produced an investment brief for a stock
(financial data, news sentiment, risk, DCF valuation, report writer).

---

## What You'll Learn

- How to build a **multi-agent pipeline** with the OpenAI Agents SDK
- The **agents-as-tools** pattern (vs. handoffs) for keeping an Orchestrator in control
- How to run **parallel** vs **sequential** sub-agent phases
- How to wire pure Python functions as tools via `@function_tool`
- Why the numbers (KPIs, CLV, verdicts) are computed in **Python**, and the decision
  rule is a **fixed table**, while the LLM interprets and writes

---

## Architecture

```
OrchestratorAgent
  Phase 1 (parallel):   customer_profile_analysis + voice_of_customer_analysis + churn_risk_assessment
  Phase 2 (after P1):   customer_value_analysis   (needs tier, spend and margin context)
  Phase 3 (last):       write_case_resolution     (synthesises everything)
```

| Agent | Tools it owns | Finance-edition equivalent |
|---|---|---|
| CustomerProfileAgent | `get_customer_profile`, `get_order_history`, `get_ticket_history`, `compute_customer_kpis` | FinancialDataAgent |
| VoiceOfCustomerAgent | `search_customer_interactions`, `analyze_sentiment` | NewsAndSentimentAgent |
| ChurnRiskAgent | `get_ticket_history`, `compute_customer_kpis`, `search_customer_interactions` | RiskAgent |
| CustomerValueAgent | `run_clv_model`, `get_segment_benchmarks` | ValuationAgent (DCF + peer multiples) |
| ResolutionWriterAgent | *(none: pure synthesis)* | ReportWriterAgent |

**CLV is a DCF of a customer.** A stock is worth its discounted future cash flows;
a customer is worth the discounted margin they generate while they stay. The
retention rate plays the role of survival probability each year.

**The service-tier table** (value verdict x churn-risk verdict, with SAFETY and
POLICY ABUSE overrides) replaces the finance edition's BUY/HOLD/SELL rating table.

---

## Sample customers (synthetic NovaMart CRM in `tools/mock_crm.py`)

| ID | Customer | Situation | Expected outcome |
|---|---|---|---|
| C1001 | Priya Sharma, Platinum, 7 yrs | Refrigerator failed twice after repair; Twitter post; consumer-court threat | VIP RECOVERY |
| C1002 | Rahul Verma, Gold | Phone delivery 8+ days late, polite but frustrated | PRIORITY CARE or STANDARD CARE |
| C1003 | Ananya Iyer, Silver | Happy, low-value customer, filter kit out for delivery | STANDARD CARE |
| C1004 | Vikram Singh, Silver | Fourth "damaged" claim in 9 months, refund to a different card, negative margin | POLICY ONLY |
| C1005 | Meera Nair, Gold | Smoke and burning smell from a new AC | SAFETY ESCALATION |

---

## Setup

```bash
cd chapter_4_lab_case_resolution_agent
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # then put your OpenAI key in .env
```

All commands below are run from **inside this folder**.

## Running the Lab

```bash
python main.py C1001
```

| Flag | What it does |
|---|---|
| `--queue ID ...` | Analyse several cases sequentially (e.g. `--queue C1002 C1005`) |
| `--message "..."` | Pass the customer's latest message into the analysis |
| `--output FILE` | Save the brief(s) to a text file |
| `--stream` | Stream tokens as they arrive |
| `--verbose` | Print every sub-agent's output |
| `--trace` | Print the raw CALL/OUTPUT tool sequence (proves parallel execution) |
| `--log-dir DIR` | Save each sub-agent's output to its own `.txt` file inside `DIR` |
| `--working-memory [FILE]` | Mirror everything printed to the terminal into a file |

```bash
python main.py C1001 --verbose --trace --log-dir logs
python main.py C1001 --queue C1002 C1003 C1004 C1005 --output briefs.txt
```

## Project Layout

```
chapter_4_lab_case_resolution_agent/
├── README.md
├── requirements.txt
├── .env.example
├── main.py                        <- CLI entry point + analyze_case / analyze_queue
├── specialist_agents/             <- agent definitions (instructions + tool wiring)
│   ├── config.py                  <- model name (CC_OPENAI_MODEL, default gpt-4o)
│   ├── orchestrator.py
│   ├── customer_profile_agent.py
│   ├── voice_of_customer_agent.py
│   ├── churn_risk_agent.py
│   ├── customer_value_agent.py
│   └── resolution_writer_agent.py
└── tools/                         <- pure Python @function_tool wrappers
    ├── mock_crm.py                <- synthetic customers, orders, tickets, interactions, tier benchmarks
    ├── cache.py
    ├── crm_tools.py
    ├── value_tools.py
    └── voc_tools.py
```

The package is called `specialist_agents/` (not `agents/`) on purpose: the OpenAI
Agents SDK is imported as `agents`, and a local folder with that name would shadow it.

---

## Lab Exercises (Suggested)

1. **Trace parallelism.** Run with `--trace`: the three Phase 1 agents should all be
   `CALL`ed before any `OUTPUT` appears.
2. **Change the decision table.** In `resolution_writer_agent.py`, make HIGH VALUE x
   MODERATE RISK a VIP RECOVERY case. Re-run C1002. Who in the business should own that table?
3. **Tune the CLV.** Change the baseline discount rate (12%) or the tier retention rates
   in `customer_value_agent.py` and `tools/mock_crm.py`. How does C1005's value verdict move?
4. **Add a sixth agent.** For example a `WarrantyPolicyAgent` that checks which remedies
   (repair, replacement, refund) the policy allows for each order. Wire it into Phase 1.
5. **Swap the model.** Set `CC_OPENAI_MODEL=gpt-4o-mini` and compare the briefs' quality.

## Disclaimer

Educational use only. All customers and data are fictional. In a real deployment a
human case owner reviews every commitment before it reaches a customer, and
customer-value segmentation must be checked for fairness (Chapter 11).
