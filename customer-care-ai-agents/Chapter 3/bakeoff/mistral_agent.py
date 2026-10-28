"""Bake-off: Mistral (function calling via la Plateforme).

Requires MISTRAL_API_KEY. Mistral's Agents API adds hosted agents with
built-in connectors (code execution, web search, document library) and
handoffs; this implementation uses plain function calling with a manual
loop, the portable subset every Mistral model supports.
"""

import json
import os

from mistralai.client import Mistral

from common import (
    compute_contact_rate_tool_func, describe, get_support_data_tool_func,
    input_message, persist_framework_output, system_message,
)

MODEL = os.getenv("BAKEOFF_MISTRAL_MODEL", "mistral-medium-latest")
MAX_TURNS = 6

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_support_data",
            "description": "Get delivered orders and support tickets for a product line",
            "parameters": {
                "type": "object",
                "properties": {"product_line": {"type": "string"}},
                "required": ["product_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_contact_rate",
            "description": "Compute contact rate (tickets per 100 orders)",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickets": {"type": "integer"},
                    "orders": {"type": "integer"},
                },
                "required": ["tickets", "orders"],
            },
        },
    },
]


def _dispatch(name: str, args: dict) -> str:
    if name == "get_support_data":
        return describe(get_support_data_tool_func(args["product_line"]))
    if name == "compute_contact_rate":
        return str(compute_contact_rate_tool_func(args["tickets"], args["orders"]))
    raise ValueError(name)


def main() -> None:
    if not os.getenv("MISTRAL_API_KEY"):
        raise SystemExit("MISTRAL_API_KEY is not set; skipping live run.")
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_message},
    ]
    in_tok = out_tok = 0
    for _ in range(MAX_TURNS):            # loop budget: never let a loop run unbounded
        resp = client.chat.complete(
            model=MODEL, messages=messages, tools=TOOLS)
        usage = resp.usage
        in_tok += usage.prompt_tokens
        out_tok += usage.completion_tokens
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            break
        for call in msg.tool_calls:
            result = _dispatch(
                call.function.name, json.loads(call.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": result,
            })
    final = msg.content
    usage_s = f"input={in_tok} output={out_tok} total={in_tok + out_tok}"
    print(final)
    print(f"\n[token usage] {usage_s}")
    persist_framework_output("mistral", str(final), usage_s)


if __name__ == "__main__":
    main()
