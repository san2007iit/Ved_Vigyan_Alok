# Chapter 10 - Agentic RAG over Customer-Experience Documents (customer-care edition)

Builds, step by step, from a naive retrieve-then-generate pipeline to an **agentic RAG**
system that routes across different CX documents, breaks comparative questions into
sub-questions, and corrects itself by re-querying when the evidence is not enough. It then
adds hybrid search, reranking and answer evaluation.

## The business question

NovaMart wants to launch a **WhatsApp support channel** this quarter. Its internal
**Service Quality Policy** says a new channel may launch only if first-contact resolution
(FCR) is at least 70%; 65-70% is *amber* (launch only with an approved remediation plan);
below 65% is a *hard block*. The **annual CX report** says FCR was **71%** at year-end
2025. The **Q3 review call** says it **dipped to 68%**. Which figure applies, and what does
the policy say? Answering needs facts from *two or three documents at once*, which is
exactly where naive RAG breaks down.

Shared question run through every retrieval pattern:
*"Does NovaMart's Q3 first-contact resolution satisfy the service-quality policy?"*
(Expected answer: no, it is amber at 68%. The launch needs a remediation plan, and the 3.7-hour
enterprise first response must be flagged because it is within 10% of the 4-hour SLA.)

## Finance edition -> customer-care edition

| Finance | Customer care (`data/`) |
|---|---|
| `acme_10k_excerpt.md` (annual report, year-end quick ratio 1.23) | `novamart_annual_cx_report_2025.md` (year-end FCR 71%, first response 3.2 h) |
| `acme_earnings_call_q3.md` (Q3 quick ratio 1.18) | `novamart_q3_cx_review_call.md` (Q3 FCR 68%, first response 3.7 h) |
| `firm_risk_policy.md` (quick ratio >= 1.00 hard block, covenant awareness) | `service_quality_policy.md` (FCR >= 70%, amber 65-70%, SLA awareness, vendor concentration) |
| "Can we open a long position in Acme?" | "Can we launch the WhatsApp channel?" |

## Steps

| Step | Pattern | What to watch |
|---|---|---|
| 3 | Naive RAG (one index, top-k=3) | Fine for "how many contacts in 2025?", shaky on the cross-document question |
| 4 | Router (`RouterQueryEngine` + `LLMSingleSelector`) | Picks ONE source; cannot combine the Q3 figure with the policy |
| 5 | Sub-question decomposition (`SubQuestionQueryEngine`) | Splits the question into "Q3 FCR?" + "policy threshold?" and combines them |
| 6 | Self-correcting agent (`FunctionAgent` over the document tools) | Decides which tools to call and re-queries until it has both the figure and the threshold |
| 7 | Hybrid dense + BM25 retrieval (reciprocal rank fusion) | Exact terms such as "CX-POL-007" or "FCR" are found even when embeddings miss them |
| 8 | Reranking (`LLMRerank`) | A broad top-6 is trimmed to the best 2 chunks |
| 9 | Hybrid + rerank | Recall first, precision second |
| 10 | Evaluation triad | Faithfulness (grounded?), answer relevancy, context relevancy |

## Tech stack

- **LlamaIndex** for indexing, routing, decomposition and the agent loop
- **Claude** for generation and reasoning via `llama-index-llms-anthropic`
  (default `claude-opus-5-5`, override with `CC_ANTHROPIC_MODEL`; the wrapper drops
  `temperature` for models that reject it)
- **OpenAI `text-embedding-3-small`** for the vector index

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # set ANTHROPIC_API_KEY and OPENAI_API_KEY
python agentic_rag_lab.py
```

`requirements.txt` pins `anthropic<2`: `llama-index-llms-anthropic` 0.12.x fails on every
chat call with anthropic 2.x (`TypeError: Cannot instantiate typing.Union`), because
`MessageParam` became a union type. Remove the pin once the wrapper supports anthropic 2.

Run a single step:

```bash
python -c "from agentic_rag_lab import load_indices, step_6_agent; step_6_agent(load_indices())"
python -c "from agentic_rag_lab import step_3_naive_rag; step_3_naive_rag()"
```

Every step prints to the terminal and writes a markdown report to `results/lab_runs/`
with the **retrieved chunks** (source file and score) and the **answer**. A full run also
writes `00_comparison_fcr_policy.md`, the side-by-side view of every pattern's answer to
the shared question. Each run makes live, paid API calls (the documents are small, so it
is cheap).

## Exercises

1. Edit the Q3 call so FCR is 63%. Which steps now report a *hard block*?
2. Delete the sentence "the latest quarterly review if it is more recent" from the policy.
   Does the agent switch to the 71% annual figure? Why is the wording of the policy itself
   a design decision for RAG?
3. Ask the agent about vendor concentration (CallBridge at 45%). Which documents does it need?
4. Add a fourth document (for example a WhatsApp launch plan) and a fourth tool. What must the
   tool description say so the router picks it?
