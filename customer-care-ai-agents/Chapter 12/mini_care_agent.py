"""mini_care_agent.py: The agent harness, by hand, on the raw OpenAI chat-completions API.
(Customer-care edition of the finance chapter's mini_analyst.py.)

The labs in this chapter run on the OpenAI Agents SDK, whose Runner is a production harness
that hides the tool-execution loop from you. This file exists for the opposite reason: to
make that loop visible. It is a single-file case-resolution agent with a hand-rolled tool
loop written directly against `client.chat.completions.create`, no framework, so that the
harness's responsibilities (the loop, the conversation state, the tool dispatch, the
termination conditions, the iteration budget, the tool permission layer) are all in plain
sight. Read it once to understand what a harness is; then let the Agents SDK provide one.

The agent answers one question: what should NovaMart do about this customer's open case?
It uses two read-only tools over the in-memory order system and inbox from care_workflow.py,
and ends its answer with a machine-readable trailer:

    RESOLUTION: REPLACE | REPAIR | REFUND | COMPENSATE | NO_ACTION | SAFETY_ESCALATION | INSUFFICIENT_EVIDENCE
    CONFIDENCE: LOW | MEDIUM | HIGH
"""

from __future__ import annotations

import json

from dotenv import load_dotenv
from openai import OpenAI

from care_workflow import ALLOWED_CUSTOMERS, MESSAGES, ORDERS

load_dotenv()

MODEL = "gpt-4o-mini"                 # the SLM tier
MAX_ITERATIONS = 6                    # loop budget: a risk limit, not a nicety
MAX_TOKENS = 700

SYSTEM_PROMPT = """You are a customer-care case assistant at NovaMart, an electronics and \
appliance retailer. Given a customer ID, recommend how to resolve their open case, using ONLY \
the tools provided. Gather the order status and the customer's messages before concluding.

Any mention of smoke, fire, burning or electric shock means SAFETY_ESCALATION. If the tools do \
not return enough evidence to judge, do not guess -- return INSUFFICIENT_EVIDENCE. Never \
promise money or outcomes: you recommend, a person approves.

End every response with exactly two trailer lines and nothing after them:
RESOLUTION: <REPLACE|REPAIR|REFUND|COMPENSATE|NO_ACTION|SAFETY_ESCALATION|INSUFFICIENT_EVIDENCE>
CONFIDENCE: <LOW|MEDIUM|HIGH>"""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_status",
        "description": "Get the customer's latest order: product, value, delivery status and "
                       "any open ticket. Use this first.",
        "parameters": {"type": "object",
                       "properties": {"customer_id": {"type": "string", "description": "e.g. C1001"}},
                       "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "get_recent_messages",
        "description": "Get the customer's most recent messages, to understand what they want.",
        "parameters": {"type": "object",
                       "properties": {"customer_id": {"type": "string", "description": "e.g. C1001"}},
                       "required": ["customer_id"]}}},
]


# --- Tool implementations (read-only; safe to retry) -------------------------------------
def _check_customer(customer_id: str) -> str:
    """Enforce the desk's allow-list before any tool touches a customer record."""
    c = (customer_id or "").strip().upper()
    if c not in ALLOWED_CUSTOMERS:
        raise PermissionError(f"customer {c!r} is not on this desk's allow-list")
    return c


def get_order_status(customer_id: str) -> dict:
    c = _check_customer(customer_id)
    return {"customer_id": c, **ORDERS[c]}


def get_recent_messages(customer_id: str, limit: int = 5) -> dict:
    c = _check_customer(customer_id)
    return {"customer_id": c, "messages": MESSAGES.get(c, [])[:limit]}


TOOL_IMPLS = {"get_order_status": get_order_status, "get_recent_messages": get_recent_messages}


def run_tool(name: str, arguments: dict) -> str:
    """Execute one tool call. A failed tool still returns a result string, so the model can
    recover -- dropping it would leave a dangling tool call and break the next turn."""
    try:
        return json.dumps(TOOL_IMPLS[name](**arguments))
    except Exception as exc:  # noqa: BLE001 -- surface any failure back to the model
        return json.dumps({"error": str(exc)})


# --- The hand-rolled agentic loop --------------------------------------------------------
def resolve(customer_id: str, client: OpenAI | None = None) -> dict:
    """Run the agent on one case. This IS the harness: it owns the conversation state, the
    tool-execution loop, the termination conditions, and the iteration budget."""
    client = client or OpenAI()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Resolve the open case for customer {customer_id}."}]

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model=MODEL, max_tokens=MAX_TOKENS, tools=TOOLS, messages=messages,
        )
        message = response.choices[0].message
        messages.append(message)

        # Termination condition: the model stopped asking for tools.
        if not message.tool_calls:
            text = message.content or ""
            return {"customer_id": customer_id, "iterations": iteration, "answer": text,
                    "resolution": _parse_trailer(text, "RESOLUTION"),
                    "confidence": _parse_trailer(text, "CONFIDENCE"), "stopped": "end_turn"}

        # Otherwise execute every requested tool and feed the results back, one per call.
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = run_tool(call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Budget-exhausted termination: the loop hit MAX_ITERATIONS without concluding.
    return {"customer_id": customer_id, "iterations": MAX_ITERATIONS, "answer": "",
            "resolution": "INSUFFICIENT_EVIDENCE", "confidence": "LOW", "stopped": "max_iterations"}


def _parse_trailer(text: str, label: str) -> str:
    """Pull a machine-readable trailer value (RESOLUTION / CONFIDENCE) out of the answer."""
    for line in reversed((text or "").splitlines()):
        if line.strip().upper().startswith(f"{label}:"):
            return line.split(":", 1)[1].strip().upper()
    return "UNKNOWN"


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    customer_id = sys.argv[1] if len(sys.argv) > 1 else "C1001"
    result = resolve(customer_id)
    print(f"[{result['customer_id']}] resolution={result['resolution']} "
          f"confidence={result['confidence']} iterations={result['iterations']} "
          f"stopped={result['stopped']}")
    print("-" * 60)
    print(result["answer"])
