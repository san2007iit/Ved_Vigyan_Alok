# Chapter 7 - Designing Multi-Agent Systems (customer-care edition)

Five runnable labs on one task: **assess the health of a NovaMart B2B service
account** (hotel chains, school trusts, clinics, co-working spaces, grocery chains
that buy appliances plus a service contract) and decide whether the relationship is
healthy.

| Finance edition | Customer-care edition |
|---|---|
| Profitability ratios (ROA, ROE, net margin, gross margin) | **Service quality**: CSAT, first-contact resolution, SLA compliance, self-service resolution |
| Liquidity ratios (current, quick, D/E, interest coverage) | **Relationship risk**: repeat-contact rate, escalation rate, first-response hours, renewal probability |
| Stock price / company info / income statement agents | **SLA performance / account profile / billing** agents |
| AAPL, MSFT, NVDA, GOOGL, JPM | AC-HOTELS, AC-SCHOOLS, AC-CLINICS, AC-COWORK, AC-RETAIL |

## Labs

| Lab | Notebook | Architecture |
|---|---|---|
| Lab 1 | `chapter_7_lab_1_multi_agent_llamaindex.ipynb` | Four-agent **sequential pipeline** (AccountData -> ServiceQuality -> RelationshipRisk -> Supervisor) with explicit handoffs and shared state (LlamaIndex `AgentWorkflow` + `Context`) |
| Lab 2 | `chapter_7_lab_2_multi_agent_openai_sdk.ipynb` | **Leader-follower routing** (OpenAI Agents SDK): a triage manager hands requests to SLA / Account Profile / Billing specialists; plus an agents-as-tools variant that synthesises all three |
| Lab 3 | `chapter_7_lab_3_multi_agent_autogen.ipynb` | **Conversational analyst** (AutoGen 0.2 API via AG2): an assistant writes pandas code on a synthetic Q3 ticket export, an executor runs it, a plot is saved |
| Lab 4 | `chapter_7_lab_4_evaluator_optimizer.ipynb` | **Evaluator-optimizer** extension: a fifth agent runs deterministic completeness checks and sends the Supervisor back for one revision |
| Lab 5 | `chapter_7_lab_5_parallel_multi_account.ipynb` | **Parallel** analysis of five accounts with `asyncio.gather`, compared with a sequential run |

Labs 1, 2, 4 and 5 use the stub data in `common.py`; the only key needed is
`OPENAI_API_KEY`. Lab 3 generates its own synthetic ticket CSV.

## Running on Google Colab

1. Open the notebook in Colab (File -> Open notebook -> GitHub, or upload it).
2. Upload `common.py` from this folder with the Files panel (Labs 1, 2, 4, 5), or set
   `CC_RAW_BASE` to the raw URL of your copy of the repository (see the bootstrap cell).
3. Add `OPENAI_API_KEY` under the key icon and enable notebook access.
4. Runtime -> Run all.

## Notes

* **Iteration limits**: four- and five-agent LlamaIndex pipelines need more workflow
  iterations than the default 20; the notebooks use `max_iterations=60`, keep every agent
  to one tool call, and make the last agent write the final answer instead of handing off.
* **AutoGen version**: the finance edition installed `pyautogen<0.4`. On current Python
  versions that pin resolves to an obsolete release, so Lab 3 installs **AG2**
  (`ag2[openai]<1.0`), the community continuation of the AutoGen 0.2 API (same imports).
  AG2 1.x introduced a new API.
* **Code execution safety**: Lab 3 runs model-written code on your machine in a temporary
  folder. Never point such an agent at production data or credentials.
