# Chapter 3 - AI Agent Frameworks (customer-care edition)

One reference task, nine frameworks. The task: **compare two NovaMart
product lines on contact rate** (support tickets per 100 delivered orders)
and write a memo for the Head of Customer Care.

| Product line | Orders (Q3) | Tickets (Q3) | Contact rate |
|---|---|---|---|
| ZX1 - Zenith X1 Smartphone | 18,400 | 2,760 | 15.0 |
| FL-340 - FrostLine Refrigerator 340L | 3,150 | 1,134 | 36.0 |

## Notebooks

| Lab | Notebook | Framework | Credential |
|---|---|---|---|
| Lab 1 | `chapter_3_lab_1_langchain.ipynb` | LangChain `create_agent` | `OPENAI_API_KEY` |
| Lab 2 | `chapter_3_lab_2_google_adk.ipynb` | Google ADK | `GOOGLE_API_KEY` |
| Lab 3 | `chapter_3_lab_3_crewai.ipynb` | CrewAI | `OPENAI_API_KEY` |
| Lab 4 | `chapter_3_lab_4_autogen.ipynb` | AutoGen AgentChat | `OPENAI_API_KEY` |
| Lab 5 | `chapter_3_lab_5_openai_sdk.ipynb` | OpenAI Agents SDK | `OPENAI_API_KEY` |
| Lab 6 | `chapter_3_lab_6_llamaindex.ipynb` | LlamaIndex `FunctionAgent` | `OPENAI_API_KEY` |
| Lab 7 | `chapter_3_lab_7_claude_no_tools_baseline.ipynb` | Anthropic API, **no tools** (baseline) | `ANTHROPIC_API_KEY` |
| Lab 8 | `chapter_3_lab_8_mistral_manual_loop.ipynb` | Mistral, **hand-written tool loop** | `MISTRAL_API_KEY` |
| Lab 9 | `chapter_3_lab_9_pydantic_ai.ipynb` | PydanticAI (typed output) | `OPENAI_API_KEY` |

The notebooks import `common.py` (task, tools, data, prompts). If you open a
notebook in Colab without cloning the repository, upload `common.py` with
the Files panel first (or set `CC_RAW_BASE`, see the bootstrap cell).

Use a separate environment for CrewAI and Google ADK: both have heavy
dependency trees.

## Also in this folder

* `bakeoff/` - the same task as nine standalone scripts that also record token
  usage to `bakeoff/outputs/` (see `bakeoff/README.md`).
* `templates/` - the nine-dimension framework decision matrix with weights tuned
  for a customer-care organisation, and `score_frameworks.py` to rank frameworks.
