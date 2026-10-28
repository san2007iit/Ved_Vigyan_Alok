# Chapter 2 - Design Patterns for AI Agents (customer-care edition)

Six notebooks, one per design pattern, all applied to NovaMart customer care.

| Lab | Notebook | Pattern | Customer-care use case | Finance-edition equivalent |
|---|---|---|---|---|
| Lab 1 | `chapter_2_lab_1_retrieving_customer_kpis_tool_use.ipynb` | **Tool use** | Agent retrieves a Platinum customer's value, service and loyalty KPIs by year; shows how the knowledge cutoff decides the "most recent year" | Apple fundamental ratios |
| Lab 2 | `chapter_2_lab_2_care_conversation_working_memory.ipynb` | **Working memory** | Six-turn chat about a failed refrigerator; the agent must remember prices, dates and credits to compute the upgrade payment (Rs 71,490 - 62,990 - 3,000 = Rs 5,500) | Portfolio rebalancing over six turns |
| Lab 3 | `chapter_2_lab_3_care_conversation_episodic_memory.ipynb` | **Episodic memory** | The chat is condensed into a JSON episode (LangChain and Responses API), stored in ChromaDB and retrieved when the customer returns a week later | Rebalancing episodes in ChromaDB |
| Lab 4 | `chapter_2_lab_4_ReAct_agent_LlamaIndex_shipment_tracking.ipynb` | **ReAct** | LlamaIndex `ReActAgent` tracks a shipment; Thought / Action / Observation trace; follow-up answered from memory | Latest stock price via Yahoo Finance |
| Lab 5 | `chapter_2_lab_5_complaint_reply_evaluator_optimizer_pattern.ipynb` | **Evaluator-optimizer** | A reply drafter writes complaint responses from a policy knowledge base; a QA evaluator checks KB citations, dated commitments, authority limits and empathy, looping until it passes (max 4 rounds) | Reuters news search + judge |
| Lab 6 | `chapter_2_lab_6_retention_decision_sequential_pattern.ipynb` | **Sequential workflow** | Retrieval -> analysis -> retention decision **INVEST / MAINTAIN / LIMIT** with a watchlist | BUY / HOLD / SELL |

Labs 2 and 3 share the same agent and conversation. Lab 3 is the fuller
treatment of long-term memory; if you only run one, run Lab 3.

## API keys

All six labs need only `OPENAI_API_KEY` (chat completions and, for Lab 3,
embeddings). All data is mock CRM / order / policy data defined in the notebooks.

## What actually persists

- `SQLiteSession("name")` keeps the conversation **in memory only**. Pass a
  file to keep it across runs: `SQLiteSession("name", "care.db")`.
- ChromaDB `PersistentClient(path="./chroma_care_episodes")` writes a folder
  next to the notebook. In Colab that folder disappears when the runtime is
  recycled; mount Google Drive if you want episodes to survive.
- The final "All in" loop in Lab 3 reads from `input()`. It stores nothing
  until you type `exit`, which triggers the episode summary and the write to ChromaDB.

## Running locally

Each notebook installs its own packages in its first cell. Copy
`../.env.sample` to `.env`, add your key, and open the notebook in Jupyter
or VS Code.
