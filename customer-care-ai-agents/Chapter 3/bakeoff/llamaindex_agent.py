"""Bake-off: LlamaIndex FunctionAgent (data-centric agent design)."""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

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
    agent = FunctionAgent(
        name="CXAnalyst",
        description="Computes and compares contact rates",
        system_prompt=system_message,
        tools=[get_support_data, compute_contact_rate],
        llm=OpenAI(model=OPENAI_MODEL),
    )
    response = await agent.run(user_msg=input_message)
    final = str(response)
    print(final)
    persist_framework_output("llamaindex", final)


if __name__ == "__main__":
    asyncio.run(main())
