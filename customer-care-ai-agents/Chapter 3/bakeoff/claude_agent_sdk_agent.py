"""Bake-off: Claude Agent SDK (safe-by-design agentic workflows).

The Claude Agent SDK drives Claude models with sessions, subagents, hooks,
and native MCP support. Custom tools are exposed to the agent as an
in-process MCP server. Requires ANTHROPIC_API_KEY with available credits
(and Node.js, because the SDK launches the Claude Code CLI under the hood).
"""

import asyncio
import os

from claude_agent_sdk import (
    ClaudeAgentOptions, create_sdk_mcp_server, query, tool,
)

from common import (
    compute_contact_rate_tool_func, describe, get_support_data_tool_func,
    input_message, persist_framework_output, system_message,
)

MODEL = os.getenv("BAKEOFF_CLAUDE_MODEL", "claude-opus-5-5")


@tool("get_support_data", "Get delivered orders and support tickets for a product line",
      {"product_line": str})
async def get_support_data(args):
    d = get_support_data_tool_func(args["product_line"])
    return {"content": [{"type": "text", "text": describe(d)}]}


@tool("compute_contact_rate", "Compute contact rate (tickets per 100 orders)",
      {"tickets": int, "orders": int})
async def compute_contact_rate(args):
    rate = compute_contact_rate_tool_func(args["tickets"], args["orders"])
    return {"content": [{"type": "text", "text": str(rate)}]}


async def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; skipping live run.")
    care_tools = create_sdk_mcp_server(
        name="care", tools=[get_support_data, compute_contact_rate])
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=system_message,
        mcp_servers={"care": care_tools},
        allowed_tools=[
            "mcp__care__get_support_data",
            "mcp__care__compute_contact_rate",
        ],
        max_turns=8,
    )
    final = []
    async for message in query(prompt=input_message, options=options):
        text = getattr(message, "result", None)
        if text:
            final.append(text)
    output = "\n".join(final)
    print(output)
    persist_framework_output("claude_agent_sdk", output)


if __name__ == "__main__":
    asyncio.run(main())
