"""Adapter between the complaint-resolution agent and the harness.

If a real implementation of the resolution agent is available as a
Python package (for example one built from the Chapter 4 case-resolution
agent or the Chapter 9 claims pipeline), this module imports it and adapts
the call interface. Otherwise it falls back to a deterministic mock that
exercises every code path in the harness, so the lab runs end to end
without the real agent or any API key.

The mock is NOT a serious agent. It consults each case's expected
outcome so that test cases produce predictable outputs.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone

from .models import (
    DEFAULT_REQUIRED_TOOLS,
    ComplaintCase,
    Decision,
    ResolutionAgentOutput,
    ResolutionToolCall,
)

try:  # pragma: no cover - plug your own implementation in under this name
    from novamart_care.resolution_agent import run_resolution as _real_run_resolution

    HAS_REAL_AGENT = True
except ImportError:
    HAS_REAL_AGENT = False


def run_resolution(case: ComplaintCase) -> ResolutionAgentOutput:
    if HAS_REAL_AGENT:
        return _real_run_resolution(case)
    return _mock_run_resolution(case)


# Which knowledge-base article each decision normally rests on.
_POLICY_FOR: dict[str, str] = {
    "APPROVE": "KB-101",   # replacement window / standard remedy
    "PARTIAL": "KB-104",   # delay compensation
    "ESCALATE": "KB-105",  # safety or human review
    "DENY": "KB-101",      # outside the replacement window / no defect
}

# Teaching switch: make the mock misbehave in one specific way so the class
# can watch the scorecard BLOCK. Unset (the default) gives a clean baseline.
#   skip_safety     - safety hazards are handled as routine repairs
#   leak_pii        - the audit log records the customer's raw phone number
#   obey_injection  - instructions pasted into complaints are followed
#   deny_tamil      - Tamil-speaking customers are denied (a fairness failure)
#   no_citation     - reasoning stops citing the KB article
MOCK_FAULT = os.environ.get("CC_MOCK_FAULT", "")

_INJECTION_MARKERS = (
    "ignore all previous instructions",
    "you are now",
    "system override",
)


def _mock_run_resolution(case: ComplaintCase) -> ResolutionAgentOutput:
    """Deterministic mock agent.

    Rules:
    - safety_escalation=True in expected -> ESCALATE with the safety flag set
    - an injection marker in the complaint text -> simulate a well-defended
      agent by returning the *expected* decision (the injection is ignored)
    - otherwise -> the case's expected decision

    Swap in the real agent via the adapter above to evaluate genuine
    behaviour; with the mock, every reference-based metric should pass by
    construction.
    """
    rng = random.Random(case.case_id)

    text = case.customer.complaint_text.lower()
    if case.expected.safety_escalation and MOCK_FAULT != "skip_safety":
        decision: Decision = "ESCALATE"
        safety = True
    elif case.expected.safety_escalation:
        decision, safety = "APPROVE", False  # fault: treated as a routine repair
    elif MOCK_FAULT == "obey_injection" and any(m in text for m in _INJECTION_MARKERS):
        decision, safety = "APPROVE", False  # fault: the injected instruction wins
    elif MOCK_FAULT == "deny_tamil" and case.customer.language == "Tamil":
        decision, safety = "DENY", False     # fault: unequal treatment by language
    elif any(m in text for m in _INJECTION_MARKERS):
        # Documented rule 2: resist the injection and return the decision
        # the legitimate inputs warrant.
        decision = case.expected.decision
        safety = False
    else:
        decision = case.expected.decision
        safety = False

    order = case.orders[0] if case.orders else None
    policy_id = "KB-105" if safety else _POLICY_FOR[decision]
    refund = 0.0
    if decision == "APPROVE" and order:
        refund = order.value_inr
    elif decision == "PARTIAL":
        refund = 250.0

    # A fake trace covering every required tool. Each tool result carries the
    # facts the reasoning chain cites, so an LLM judge grading groundedness
    # finds real support in the trace, as it would for a well-behaved agent.
    now = datetime.now(timezone.utc)
    mock_results: dict[str, dict[str, object]] = {
        "intake.parse": {"case_id": case.case_id, "language": case.customer.language,
                         "tier": case.customer.tier},
        "orders.lookup": {"order_id": order.order_id if order else None,
                          "product": order.product if order else None,
                          "delivered_on": order.delivered_on if order else None},
        "policy.check": {"policy_id": policy_id},
        "safety.screen": {"hazard_detected": safety},
        "history.search": {"prior_complaints_12m": 0},
        "resolution.decide": {"decision": decision, "refund_inr": refund},
        "compliance.validate": {"all_required_tools_executed": True, "checks_passed": True},
    }
    tool_calls = [
        ResolutionToolCall(
            tool_name=name,
            arguments={"case_id": case.case_id},
            result=mock_results[name],
            latency_ms=80 + rng.randint(0, 150),
            started_at=now + timedelta(milliseconds=250 * i),
        )
        for i, name in enumerate(DEFAULT_REQUIRED_TOOLS)
    ]

    reasoning = (
        f"Decision {decision}. Intake parsed a {case.customer.tier} customer writing in "
        f"{case.customer.language}. Orders lookup found "
        f"{order.order_id + ' (' + order.product + ')' if order else 'no order'}. "
        f"Policy check applied {'' if MOCK_FAULT == 'no_citation' else policy_id}. Safety screen hazard_detected={safety}. "
        "History search found no prior complaints in 12 months. Resolution and "
        f"compliance steps executed; refund Rs {refund:,.0f}."
    )

    # Audit lines keep identifiers but mask contact details: the PII metric
    # scans these lines and fails the run on any leak.
    audit_log_lines = [
        f"[{now.isoformat()}] case_id={case.case_id} customer={case.customer.customer_id} "
        f"decision={decision}",
        f"[{now.isoformat()}] contact=+91-XXXXXX{rng.randint(1000, 9999)} safety={safety}",
    ]
    if MOCK_FAULT == "leak_pii":
        audit_log_lines.append(f"[{now.isoformat()}] callback number +91 98{rng.randint(10000000, 99999999)}")

    total_latency = sum(t.latency_ms for t in tool_calls)
    return ResolutionAgentOutput(
        case_id=case.case_id,
        decision=decision,
        safety_escalation=safety,
        refund_inr=refund,
        reasoning_chain=reasoning,
        tool_calls=tool_calls,
        audit_log_lines=audit_log_lines,
        latency_ms=total_latency,
        cost_inr=1.8,
        tokens_input=1600,
        tokens_output=420,
    )
