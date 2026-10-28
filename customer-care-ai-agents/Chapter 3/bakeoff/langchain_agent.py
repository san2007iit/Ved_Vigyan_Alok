"""Bake-off: LangChain 1.x create_agent (graph orchestration under the hood)."""

from langchain.agents import create_agent

from common import (
    OPENAI_MODEL, check_api_key, compute_contact_rate_tool_func, describe,
    get_support_data_tool_func, input_message, persist_framework_output,
    system_message,
)


def get_support_data(product_line: str) -> str:
    """Get the quarter's delivered orders and support tickets for a product line (e.g. ZX1, FL-340)."""
    return describe(get_support_data_tool_func(product_line))


def compute_contact_rate(tickets: int, orders: int) -> float:
    """Compute the contact rate (support tickets per 100 orders)."""
    return compute_contact_rate_tool_func(tickets, orders)


def main() -> None:
    check_api_key()
    agent = create_agent(
        model=f"openai:{OPENAI_MODEL}",
        tools=[get_support_data, compute_contact_rate],
        system_prompt=system_message,
    )
    print(f"Task: {input_message}")
    print("=" * 60)
    output = agent.invoke(
        {"messages": [{"role": "user", "content": input_message}]}
    )
    in_tok = out_tok = 0
    lines = []
    for msg in output["messages"]:
        for call in getattr(msg, "tool_calls", []) or []:
            lines.append(f"[Tool: {call['name']}] args={call['args']}")
        meta = getattr(msg, "usage_metadata", None)
        if meta:
            in_tok += meta.get("input_tokens", 0)
            out_tok += meta.get("output_tokens", 0)
    final = output["messages"][-1].content
    trace = "\n".join(lines)
    usage = f"input={in_tok} output={out_tok} total={in_tok + out_tok}"
    print(trace)
    print("\n[Agent Response]\n")
    print(final)
    print(f"\n[token usage] {usage}")
    persist_framework_output("langchain", trace + "\n\n" + str(final), usage)


if __name__ == "__main__":
    main()
