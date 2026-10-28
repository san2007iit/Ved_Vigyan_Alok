"""Bake-off: PydanticAI (schema-first validation)."""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from common import (
    OPENAI_MODEL, check_api_key, compute_contact_rate_tool_func,
    get_support_data_tool_func, input_message, persist_framework_output,
    system_message,
)


class ContactRateMemo(BaseModel):
    """The validated output schema every run must satisfy."""

    zx1_contact_rate: float = Field(description="Tickets per 100 orders for ZX1")
    fl340_contact_rate: float = Field(description="Tickets per 100 orders for FL-340")
    memo: str = Field(description="Short comparative memo for the Head of Customer Care")


agent = Agent(
    f"openai:{OPENAI_MODEL}",
    system_prompt=system_message,
    output_type=ContactRateMemo,
)


@agent.tool
def get_support_data(ctx: RunContext, product_line: str) -> dict:
    """Get orders and support tickets for a product line (e.g. ZX1, FL-340)."""
    d = get_support_data_tool_func(product_line)
    return {"product_line": d.product_line, "product": d.product,
            "orders": d.orders, "tickets": d.tickets}


@agent.tool
def compute_contact_rate(ctx: RunContext, tickets: int, orders: int) -> float:
    """Compute the contact rate (support tickets per 100 orders)."""
    return compute_contact_rate_tool_func(tickets, orders)


def main() -> None:
    check_api_key()
    result = agent.run_sync(input_message)
    memo = result.output
    u = result.usage() if callable(result.usage) else result.usage
    usage = (f"input={u.input_tokens} output={u.output_tokens} "
             f"total={u.total_tokens}")
    print(memo.model_dump_json(indent=2))
    print(f"\n[token usage] {usage}")
    persist_framework_output(
        "pydantic_ai", memo.model_dump_json(indent=2), usage)


if __name__ == "__main__":
    main()
