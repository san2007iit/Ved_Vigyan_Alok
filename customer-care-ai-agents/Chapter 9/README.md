# Chapter 9 - Building Multi-Agent Warranty and Returns Workflows (customer-care edition)

Three runnable labs plus the chapter's project source. The finance edition built
**insurance** workflows (claims, fraud, underwriting); here the same architectures run
NovaMart's **after-sales service**: Care+ extended-warranty claims, return-abuse
investigations, and pricing Care+ plans.

| Finance (insurance) edition | Customer-care edition |
|---|---|
| Auto-collision insurance claim (Sarah Chen) | Care+ warranty claim on a water purifier (**Ananya Iyer**, C1003) |
| Policy (period, limit, deductible, exclusions) | Warranty plan (period, limit, **service fee**, exclusions) |
| FNOL, police report | Claim intake, **technician job sheet** |
| Fraud screener and fraud debate | **Return-abuse** screener and abuse debate (**Vikram Singh**, C1004) |
| Underwriting (credit, driving record, property value) | **Care+ pricing** (product reliability, claims history, service-area risk) for **Rahul Verma** (C1002) |

## Labs

| Lab | Notebook | Architecture |
|---|---|---|
| Lab 1 | `chapter_9_lab_1_warranty_claims_pipeline.ipynb` | Seven-agent **sequential pipeline** with a compliance guardrail and an audit trail (Intake -> Evidence -> Coverage -> Abuse screen -> Assessment -> Decision -> Compliance). Runs one approved claim and one denied claim. |
| Lab 2 | `chapter_9_lab_2_return_abuse_investigation.ipynb` | **Adversarial debate**: Abuse Advocate vs Customer Advocate, judged by an Investigation Decision agent that outputs a referral signal, not a verdict |
| Lab 3 | `chapter_9_lab_3_care_plus_pricing.ipynb` | **Sequential + parallel**: three enrichment calls with `asyncio.gather`, then a risk score, premium and accept/refer decision (no LLM needed) |

Labs 1 and 2 use LlamaIndex with OpenAI `gpt-4o` and need `OPENAI_API_KEY`. Lab 3 runs offline.
`common.py` holds the shared Pydantic models, sample data and helpers (`log_audit`, `days_between`).

## Running on Google Colab

1. Open the notebook in Colab and upload `common.py` from this folder (Labs 1 and 2),
   or set `CC_RAW_BASE` to the raw URL of your copy of the repository.
2. Add `OPENAI_API_KEY` under the key icon and enable notebook access.
3. Runtime -> Run all.

## Project source (`src/`)

```
src/
├── common.py, models.py                 shared configuration and data contracts
├── claims_pipeline/                     tools.py, agents.py, run.py (7-agent pipeline)
├── abuse_investigation/                 agents.py, run.py (3-agent debate)
└── warranty_underwriting/run.py         parallel enrichment + pricing
data/sample_claim.json, sample_plan.json the claim, evidence and Care+ plan used by run.py
```

From this folder:

```bash
pip install -r requirements.txt
python -m src.claims_pipeline.run
python -m src.abuse_investigation.run
python -m src.warranty_underwriting.run
```

The notebooks copy their tool code from `src/` so the two never drift apart.

## Notes

* **Iteration limits.** Seven agents each spend about three workflow iterations (tool call,
  handoff, response), so Lab 1 runs with `max_iterations=60` (the default is 20), keeps every
  agent to one tool call, and makes the compliance agent finish with a summary instead of
  handing control back up the pipeline.
* **The LLM routes; Python decides.** Coverage, abuse score, covered amount and the decision are
  deterministic functions. That is what makes a refusal explainable to the customer.
* **The compliance checks are teaching rules, not legal advice.** A real deployment maps them to
  the Consumer Protection Act, 2019 and the company's own warranty terms with its legal team.

## Exercises

1. Add a fourth red flag to `screen_for_abuse`: the serial number on the job sheet does not match
   the invoice. What weight would you give it, and why?
2. Pass `claims_history` with three earlier claims into Lab 1. Which agent changes its output, and
   what does the customer experience?
3. In Lab 2, give the Customer Advocate the technician's report as a tool. Does the determination move?
4. In Lab 3, make the premium depend on `avg_repair_cost_inr` instead of the device value. Which
   customers win and which lose?
