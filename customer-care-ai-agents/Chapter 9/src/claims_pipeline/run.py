"""
run.py
======
Entry point for the warranty claims pipeline.

Usage (from the Chapter 9 folder):
    python -m src.claims_pipeline.run

Processes the sample Care+ warranty claim (Ananya's water purifier)
through the 7-agent pipeline.
"""

import asyncio
import json
from pathlib import Path

from llama_index.core.agent.workflow import AgentWorkflow

from .agents import (
    intake_agent, parser_agent, coverage_agent,
    abuse_agent, assessment_agent, decision_agent,
    compliance_agent,
)

DATA = Path(__file__).resolve().parents[2] / "data"


def load_sample() -> tuple[dict, dict, list[dict]]:
    claim_file = json.loads((DATA / "sample_claim.json").read_text(encoding="utf-8"))
    plan = json.loads((DATA / "sample_plan.json").read_text(encoding="utf-8"))
    return claim_file["claim"], plan, claim_file["submitted_documents"]


def flatten_plan(plan: dict) -> dict:
    """Turn the nested plan document into the flat record the tools read."""
    cov = plan["coverage"]
    return {
        "plan_number": plan["plan_number"],
        "customer_name": plan["customer"]["name"],
        "inception_date": cov["inception_date"],
        "expiration_date": cov["expiration_date"],
        "product_category": plan["product"]["category"],
        "coverage_limit": cov["coverage_limit"],
        "service_fee": cov["service_fee"],
        "coverage_sections": cov["name"],
        "exclusions": cov["exclusions"],
    }


async def run_claims_pipeline():
    """Run the full 7-agent warranty claims pipeline."""
    claim, plan, documents = load_sample()

    workflow = AgentWorkflow(
        agents=[
            intake_agent, parser_agent, coverage_agent,
            abuse_agent, assessment_agent, decision_agent,
            compliance_agent,
        ],
        root_agent=intake_agent.name,
        initial_state={
            "claim": {},
            "plan": flatten_plan(plan),
            "submitted_documents": documents,
            "claims_history": plan.get("claims_history", []),
            "intake_date": "2026-10-23",
        },
    )

    # Seven agents each need ~3 workflow iterations (tool call, handoff, response).
    handler = workflow.run(user_msg=json.dumps(claim), max_iterations=60)

    # Stream agent transitions (banner only when the agent changes)
    last_agent = None
    async for event in handler.stream_events():
        name = getattr(event, "current_agent_name", None)
        if name and name != last_agent:
            print(f"\n{'=' * 50}\n  Agent: {name}\n{'=' * 50}")
            last_agent = name
        delta = getattr(event, "delta", None)
        if delta:
            print(delta, end="", flush=True)

    result = await handler

    print(f"\n\n{'=' * 60}\n  AUDIT TRAIL\n{'=' * 60}")
    state = await handler.ctx.store.get("state")
    for entry in state["claim"].get("audit_log", []):
        print(f"  [{entry.get('timestamp', '')}] "
              f"{entry.get('agent', '')}: {entry.get('action', '')}")
    print(f"\nDecision: {state['claim'].get('decision')} | "
          f"Covered: Rs {state['claim'].get('covered_amount', 0):,.0f}")
    print(result)
    return result


if __name__ == "__main__":
    print("Processing Care+ warranty claim...\n")
    asyncio.run(run_claims_pipeline())
