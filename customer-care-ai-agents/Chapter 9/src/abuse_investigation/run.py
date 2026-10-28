"""
run.py
======
Run the return-abuse debate on the suspicious sample claim.

Usage (from the Chapter 9 folder):
    python -m src.abuse_investigation.run
"""

import asyncio
import sys
from pathlib import Path

from llama_index.core.agent.workflow import AgentWorkflow

from .agents import abuse_advocate, customer_advocate, investigation_decision

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import SUSPICIOUS_CLAIM_SUMMARY  # noqa: E402


async def run_investigation():
    workflow = AgentWorkflow(
        agents=[abuse_advocate, customer_advocate, investigation_decision],
        root_agent="AbuseAdvocate",
        initial_state={"claim_summary": SUSPICIOUS_CLAIM_SUMMARY},
    )
    user_msg = (
        "Investigate the following potentially abusive claim. Argue both sides "
        f"before reaching a determination.\n\nClaim summary:\n{SUSPICIOUS_CLAIM_SUMMARY}"
    )
    handler = workflow.run(user_msg=user_msg)
    last_agent = None
    async for event in handler.stream_events():
        name = getattr(event, "current_agent_name", None)
        if name and name != last_agent:
            print(f"\n=== {name} ===")
            last_agent = name
        delta = getattr(event, "delta", None)
        if delta:
            print(delta, end="", flush=True)
    final = await handler
    print("\n\nFinal determination:\n" + str(final))
    return final


if __name__ == "__main__":
    asyncio.run(run_investigation())
