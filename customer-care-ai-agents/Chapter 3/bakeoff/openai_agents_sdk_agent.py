"""Bake-off: OpenAI Agents SDK (sequential tool calling)."""

from agents import Agent, Runner, function_tool

from common import (
    OPENAI_MODEL, check_api_key, compute_contact_rate_tool_func, describe,
    get_support_data_tool_func, input_message, persist_framework_output,
    system_message,
)


@function_tool
def get_support_data(product_line: str) -> str:
    """Get the quarter's delivered orders and support tickets for a product line (e.g. ZX1, FL-340)."""
    return describe(get_support_data_tool_func(product_line))


@function_tool
def compute_contact_rate(tickets: int, orders: int) -> float:
    """Compute the contact rate (support tickets per 100 orders)."""
    return compute_contact_rate_tool_func(tickets, orders)


def main() -> None:
    check_api_key()
    agent = Agent(
        name="Customer Experience Analysis Agent",
        instructions=system_message,
        tools=[get_support_data, compute_contact_rate],
        model=OPENAI_MODEL,
    )
    print("Running agent...")
    result = Runner.run_sync(agent, input_message)
    final_response = (
        result.final_output if hasattr(result, "final_output")
        else str(result)
    )
    usage = ""
    try:
        u = result.context_wrapper.usage
        usage = (f"input={u.input_tokens} output={u.output_tokens} "
                 f"total={u.total_tokens}")
    except AttributeError:
        pass
    print(final_response)
    if usage:
        print(f"\n[token usage] {usage}")
    persist_framework_output("openai_agents_sdk", final_response, usage)


if __name__ == "__main__":
    main()
