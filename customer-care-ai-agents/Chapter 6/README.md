# Chapter 6 - Reasoning paradigms for AI agents (customer-care edition)

Five self-contained labs, one per reasoning paradigm. All run on Claude
(`claude-opus-5-5` by default; set `CC_ANTHROPIC_MODEL` to change it).

| Lab | Folder | Paradigm | Customer-care task | Finance-edition task |
|---|---|---|---|---|
| 1 | `chapter_6_lab_1_self-refine` | Self-Refine | Draft, critique and refine the resolution letter for an escalated complaint | Investment thesis |
| 2 | `chapter_6_lab_2_self-consistency` | Self-Consistency | Majority vote on REPAIR / REPLACE / REFUND for an ambiguous AC complaint | BUY / HOLD / SELL |
| 3 | `chapter_6_lab_3_ReAct` | ReAct (Claude Agent SDK) | Is a customer's contact pattern abnormal for their tier, and is proactive delay compensation due? | Is Apple overvalued vs its sector? |
| 4 | `chapter_6_lab_4_ToT` | Tree of Thoughts | Choose a service-recovery strategy and its implementation for a high-value customer | Hedge a concentrated position into earnings |
| 5 | `chapter_6_lab_5_lats` | LATS (tree search + reflection) | Cut the average queue wait from 40 to 30 min with staffing moves at minimum cost | Trim tech exposure from 40% to 30% |

Labs 1, 2 and 4 use the Anthropic API client (`anthropic`); labs 3 and 5 use the
Claude Agent SDK (`claude-agent-sdk`, which needs Node.js). Each folder has its own
`README.md`, `requirements.txt` and `.env.example`.

Notes on current Claude models:
* Every Anthropic API call opts into server-side **refusal fallbacks**
  (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`) and checks
  `stop_reason` before reading the answer.
* Responses can contain `thinking` blocks before the text, so the code always selects
  blocks with `type == "text"` instead of reading `content[0]`.
* Claude Opus 4.7 and later do not accept `temperature`; the self-consistency lab
  therefore relies on default sampling (see that lab's README).
