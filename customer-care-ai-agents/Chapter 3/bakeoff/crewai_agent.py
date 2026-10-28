"""Bake-off: CrewAI (role-based collaboration)."""

from crewai import Agent, Crew, Task
from crewai.tools import tool

from common import (
    OPENAI_MODEL, check_api_key, compute_contact_rate_tool_func, describe,
    get_support_data_tool_func, input_message, persist_framework_output,
    system_message,
)


@tool("Get support data")
def get_support_data(product_line: str) -> str:
    """Get the quarter's delivered orders and support tickets for a product line (e.g. ZX1, FL-340)."""
    return describe(get_support_data_tool_func(product_line))


@tool("Compute contact rate")
def compute_contact_rate(tickets: int, orders: int) -> float:
    """Compute the contact rate (support tickets per 100 orders)."""
    return compute_contact_rate_tool_func(tickets, orders)


def main() -> None:
    check_api_key()
    analyst = Agent(
        role="Customer Experience Analyst",
        goal="Compare product lines on contact rate and write a short memo",
        backstory=system_message,
        tools=[get_support_data, compute_contact_rate],
        llm=OPENAI_MODEL,
        verbose=False,
    )
    task = Task(
        description=input_message,
        expected_output="A short comparative memo with both contact rates",
        agent=analyst,
    )
    crew = Crew(agents=[analyst], tasks=[task])
    result = crew.kickoff()
    final = str(result)
    usage = ""
    metrics = getattr(crew, "usage_metrics", None)
    if metrics:
        usage = (f"input={metrics.prompt_tokens} "
                 f"output={metrics.completion_tokens} "
                 f"total={metrics.total_tokens}")
    print(final)
    if usage:
        print(f"\n[token usage] {usage}")
    persist_framework_output("crewai", final, usage)


if __name__ == "__main__":
    main()
