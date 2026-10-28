"""Bake-off: AutoGen 0.4+ AgentChat (conversational coordination)."""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from common import (
    OPENAI_MODEL, check_api_key, compute_contact_rate_tool_func, describe,
    get_support_data_tool_func, input_message, persist_framework_output,
    system_message,
)


def get_support_data(product_line: str) -> str:
    """Get the quarter's delivered orders and support tickets for a product line."""
    return describe(get_support_data_tool_func(product_line))


def compute_contact_rate(tickets: int, orders: int) -> float:
    """Compute the contact rate (support tickets per 100 orders)."""
    return compute_contact_rate_tool_func(tickets, orders)


async def main() -> None:
    check_api_key()
    model_client = OpenAIChatCompletionClient(
        model=OPENAI_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    get_support_data_tool = FunctionTool(
        get_support_data,
        description="Get orders and support tickets for a product line (e.g., ZX1, FL-340)",
    )
    compute_contact_rate_tool = FunctionTool(
        compute_contact_rate,
        description="Compute contact rate (tickets per 100 orders) given tickets and orders",
    )
    analyst_agent = AssistantAgent(
        name="CXAnalyst",
        model_client=model_client,
        tools=[get_support_data_tool, compute_contact_rate_tool],
        system_message=system_message,
        reflect_on_tool_use=True,
    )
    team = RoundRobinGroupChat(
        participants=[analyst_agent],
        termination_condition=MaxMessageTermination(10),
    )
    print(f"Running AutoGen analysis: {input_message}\n")
    result = await team.run(task=input_message)

    in_tok = out_tok = 0
    lines = []
    for msg in result.messages:
        usage = getattr(msg, "models_usage", None)
        if usage:
            in_tok += usage.prompt_tokens
            out_tok += usage.completion_tokens
        content = getattr(msg, "content", "")
        lines.append(f"[{msg.source}] {content}")
    usage_s = f"input={in_tok} output={out_tok} total={in_tok + out_tok}"
    print("\n".join(lines[1:]))
    print(f"\n[token usage] {usage_s}")
    persist_framework_output("autogen", "\n\n".join(lines[1:]), usage_s)
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
