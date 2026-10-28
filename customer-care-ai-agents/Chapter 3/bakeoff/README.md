# Chapter 3 - framework bake-off (customer-care edition)

The same customer-care "Hello World" task, compare two product lines on
**contact rate** (support tickets per 100 delivered orders) and write a memo
for the Head of Customer Care, implemented in nine frameworks. All
implementations share `common.py` (mock data, prompts, tools, output
capture), so differences in the outputs reflect the frameworks, not the inputs.

| Product line | Orders (Q3) | Tickets (Q3) | Contact rate |
|---|---|---|---|
| ZX1 - Zenith X1 Smartphone | 18,400 | 2,760 | 15.0 per 100 orders |
| FL-340 - FrostLine Refrigerator 340L | 3,150 | 1,134 | 36.0 per 100 orders |

*Finance edition*: AAPL vs JPM on P/E (price / EPS). Contact rate is the
customer-care analogue: one number normalised by another so that a
high-volume product and a low-volume product can be compared fairly.

| File | Framework | Needs |
|---|---|---|
| `openai_agents_sdk_agent.py` | OpenAI Agents SDK | `OPENAI_API_KEY` |
| `langchain_agent.py` | LangChain 1.x `create_agent` | `OPENAI_API_KEY` |
| `autogen_agent.py` | AutoGen (AgentChat) | `OPENAI_API_KEY` |
| `pydantic_ai_agent.py` | PydanticAI | `OPENAI_API_KEY` |
| `llamaindex_agent.py` | LlamaIndex `FunctionAgent` | `OPENAI_API_KEY` |
| `crewai_agent.py` | CrewAI | `OPENAI_API_KEY` |
| `google_adk_agent.py` | Google ADK | `GOOGLE_API_KEY` |
| `mistral_agent.py` | Mistral (function calling, manual loop) | `MISTRAL_API_KEY` |
| `claude_agent_sdk_agent.py` | Claude Agent SDK | `ANTHROPIC_API_KEY` + Node.js |

## Run

```bash
pip install -r requirements.txt   # crewai / google-adk: see the note in the file
export OPENAI_API_KEY=...         # plus provider keys as needed
python openai_agents_sdk_agent.py
```

Each run prints the memo plus measured token usage and writes both to
`outputs/<framework>.txt`. Compare the files side by side in class: the
numbers should be identical (15.0 and 36.0) because they come from tools;
the *wording*, token usage and amount of code differ by framework.
