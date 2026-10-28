# Chapter 1 - What Are AI Agents? (customer-care edition)

Three notebooks that build up from a plain LLM API call to a first agentic
workflow, using the same NovaMart customer-care material throughout.

| Lab | Notebook | What it covers | Finance-edition equivalent |
|---|---|---|---|
| Lab 1 | `chapter_1_lab_1_API_calls_support_call_summary.ipynb` | The same **support-call transcript** (Priya Sharma's failed refrigerator repair) summarised into a structured JSON note by OpenAI, Google Gemini, Anthropic Claude (with structured outputs) and an open-weights Llama 3.1 8B; then a reasoning model | Apple earnings-call summary |
| Lab 2 | `chapter_1_lab_2_knowledge_cutoff_and_private_data.ipynb` | The two knowledge gaps of a plain LLM: the **time gap** (training cutoff) and the **access gap** (it has never seen your orders) | Knowledge cutoff on NVIDIA news |
| Lab 3 | `chapter_1_lab_3_non-agentic_vs_agentic_workflow.ipynb` | "Where is my order ORD-78120?": a bare LLM call, then manual function calling, then agents with the OpenAI Agents SDK (order-status agent and service-alerts agent) | NVIDIA price + news agents |

All three run on Google Colab or locally. The transcript, orders and service
alerts are **synthetic** and defined inside the notebooks; no CRM or news
API is needed.

## API keys per lab

| Lab | Keys |
|---|---|
| Lab 1 | `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, and `HF_TOKEN` for the open-weights section |
| Lab 2 | `OPENAI_API_KEY` |
| Lab 3 | `OPENAI_API_KEY` |

You only need the keys for the sections you actually run; each provider
section of Lab 1 is independent. In Colab, add keys under the **key** icon
in the left sidebar; locally, put them in a `.env` file (see `../.env.sample`).

## Notes

* The Llama section of Lab 1 needs a GPU runtime (Colab: *Runtime -> Change
  runtime type -> T4 GPU*) and accepted access to the gated model card.
* Notebooks call agents with `await Runner.run(...)`: Jupyter and Colab already
  run an event loop, and current versions of the Agents SDK refuse
  `Runner.run_sync()` inside a running loop. In a `.py` script use `Runner.run_sync()`.
* The Claude cell uses `claude-opus-5-5` by default; set `CC_ANTHROPIC_MODEL`
  to use another Claude model.
