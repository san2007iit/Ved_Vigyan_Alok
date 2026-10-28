"""
agents.py
=========
Agent definitions for the warranty claims pipeline.

Architecture: Sequential pipeline with guardrails (Ch 7, Section 2.1).
Each agent has a focused system prompt, a dedicated tool, and an
explicit can_handoff_to declaring the next agent in the sequence.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from ..common import MODEL
from .tools import (
    process_intake,
    parse_evidence,
    verify_coverage,
    screen_for_abuse,
    assess_repair,
    make_decision,
    validate_compliance,
)

llm = OpenAI(model=MODEL, temperature=0)

ONCE = "Call your tool exactly once; never call it a second time. "

# --- Agent 1: Intake Agent (root) ---
intake_agent = FunctionAgent(
    name="IntakeAgent",
    description="Register the customer's warranty claim.",
    system_prompt=ONCE + (
        "You handle claim intake: call process_intake with the claim data, "
        "then hand off to EvidenceParser."
    ),
    llm=llm, tools=[process_intake], can_handoff_to=["EvidenceParser"],
)

# --- Agent 2: Evidence Parser ---
parser_agent = FunctionAgent(
    name="EvidenceParser",
    description="Parse the claim form, photos and technician job sheet.",
    system_prompt=ONCE + (
        "Parse the submitted evidence with parse_evidence, then IMMEDIATELY hand off "
        "to CoverageAgent. Never answer the customer directly."
    ),
    llm=llm, tools=[parse_evidence], can_handoff_to=["CoverageAgent"],
)

# --- Agent 3: Coverage Verification ---
coverage_agent = FunctionAgent(
    name="CoverageAgent",
    description="Verify warranty-plan coverage for the failure.",
    system_prompt=ONCE + (
        "Verify whether the failure is covered by the plan (period, limit, service "
        "fee, exclusions) with verify_coverage, then hand off to AbuseScreener."
    ),
    llm=llm, tools=[verify_coverage], can_handoff_to=["AbuseScreener"],
)

# --- Agent 4: Abuse Screening ---
abuse_agent = FunctionAgent(
    name="AbuseScreener",
    description="Screen the claim for warranty-abuse indicators.",
    system_prompt=ONCE + (
        "Score the claim against abuse red flags with screen_for_abuse, "
        "then hand off to AssessmentAgent."
    ),
    llm=llm, tools=[screen_for_abuse], can_handoff_to=["AssessmentAgent"],
)

# --- Agent 5: Repair Assessment ---
assessment_agent = FunctionAgent(
    name="AssessmentAgent",
    description="Estimate the covered repair cost.",
    system_prompt=ONCE + (
        "Estimate the covered amount with assess_repair (it applies the service fee "
        "and the plan limit), then IMMEDIATELY hand off to DecisionAgent. "
        "Never answer the customer directly."
    ),
    llm=llm, tools=[assess_repair], can_handoff_to=["DecisionAgent"],
)

# --- Agent 6: Decision ---
decision_agent = FunctionAgent(
    name="DecisionAgent",
    description="Make the claim decision.",
    system_prompt=ONCE + (
        "Record the decision (APPROVED, DENIED or ESCALATED) with its reasoning chain "
        "using make_decision, then IMMEDIATELY hand off to ComplianceAgent."
    ),
    llm=llm, tools=[make_decision], can_handoff_to=["ComplianceAgent"],
)

# --- Agent 7: Compliance (Guardrail) ---
compliance_agent = FunctionAgent(
    name="ComplianceAgent",
    description="Consumer-protection guardrail with veto power.",
    system_prompt=ONCE + (
        "Validate the decision with validate_compliance. You have VETO power. Whether "
        "the result is compliant or BLOCKED, do NOT hand off: write a short final "
        "summary of the claim decision (or the blocking issues) and stop. Hand off to "
        "IntakeAgent only if the tool reports that the claim record itself is "
        "fundamentally malformed."
    ),
    llm=llm, tools=[validate_compliance], can_handoff_to=["IntakeAgent"],
)
