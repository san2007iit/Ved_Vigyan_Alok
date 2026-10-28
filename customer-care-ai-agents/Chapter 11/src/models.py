"""Pydantic schemas for the complaint-resolution evaluation harness.

Every artifact that flows through the harness is typed here:
- ComplaintCase / ExpectedResolution — input + gold answer
- ResolutionAgentOutput / ResolutionToolCall — what the agent produces
- EvaluationResult / Scorecard — what the harness produces

Keeping these in one file makes the contract between agent, metrics, and
scorecard inspectable in a single place.

Finance (KYC) edition -> customer-care edition:
  risk tier LOW/MEDIUM/HIGH/DECLINE -> decision APPROVE/PARTIAL/ESCALATE/DENY
  sanctions match                   -> safety escalation (fire, smoke, shock, swelling battery)
  jurisdiction / documents          -> customer tier, language, region / order references
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["APPROVE", "PARTIAL", "ESCALATE", "DENY"]
Severity = Literal["catastrophic", "severe", "significant"]
Segment = Literal["core", "safety", "adversarial", "disparate_impact", "regression"]

DEFAULT_REQUIRED_TOOLS = [
    "intake.parse",
    "orders.lookup",
    "policy.check",
    "safety.screen",
    "history.search",
    "resolution.decide",
    "compliance.validate",
]


class CustomerProfile(BaseModel):
    full_name: str
    customer_id: str
    tier: Literal["Platinum", "Gold", "Silver"]
    language: str  # preferred language, e.g. "Hindi", "Tamil", "English"
    region: str  # North / South / East / West
    complaint_text: str  # the customer's own words; adversarial inputs land here
    demographic_metadata: dict[str, str] = Field(default_factory=dict)


class OrderReference(BaseModel):
    order_id: str
    product: str
    value_inr: float
    delivered_on: str | None = None  # ISO 8601


class ExpectedResolution(BaseModel):
    decision: Decision
    safety_escalation: bool
    required_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_REQUIRED_TOOLS))
    rationale: str
    severity_if_wrong: Severity = "severe"


class CaseMetadata(BaseModel):
    source: Literal["historical", "synthetic", "adversarial", "production_sample"]
    created_at: datetime
    label_version: str = "v1"
    labelers: list[str] = Field(default_factory=list)
    label_kappa: float | None = None  # inter-rater reliability, if multi-labeled


class ComplaintCase(BaseModel):
    case_id: str
    segment: Segment
    customer: CustomerProfile
    orders: list[OrderReference] = Field(default_factory=list)
    expected: ExpectedResolution
    metadata: CaseMetadata


class ResolutionToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, object]
    result: dict[str, object]
    latency_ms: int
    started_at: datetime


class ResolutionAgentOutput(BaseModel):
    case_id: str
    decision: Decision
    safety_escalation: bool
    refund_inr: float = 0.0
    reasoning_chain: str  # free-form justification shown to the case owner
    tool_calls: list[ResolutionToolCall]
    audit_log_lines: list[str] = Field(default_factory=list)
    latency_ms: int
    cost_inr: float
    tokens_input: int = 0
    tokens_output: int = 0


class MetricResult(BaseModel):
    metric_name: str
    dimension: str
    methodology: str
    value: float  # 0.0 to 1.0 (or a raw value for inverted metrics)
    threshold: float
    passed: bool
    inverted: bool = False  # raw-valued metric where lower is better
    severity: Severity
    weight: float
    notes: str = ""
    detail: dict[str, object] = Field(default_factory=dict)


class Scorecard(BaseModel):
    agent_version: str
    dataset_version: str
    methodology_version: str
    executed_at: datetime
    executor: str
    segment_filter: str = "all"
    metrics: list[MetricResult]
    weighted_score: float
    catastrophic_passes: bool
    ship_decision: Literal["SHIP", "BLOCK"]

    def block_reasons(self) -> list[str]:
        # A metric that never ran fails closed; say so instead of
        # reporting a misleading literal 0.000.
        return [
            (f"{m.metric_name} ({m.severity}): NOT MEASURED in this run "
             f"— fails closed"
             if m.notes and m.notes.startswith("NOT MEASURED")
             else f"{m.metric_name} ({m.severity}): {m.value:.3f} "
                  f"vs threshold {m.threshold:.3f}")
            for m in self.metrics
            if not m.passed and m.severity == "catastrophic"
        ]


class EvaluationResult(BaseModel):
    """Per-case result, separate from per-metric aggregate."""

    case_id: str
    output: ResolutionAgentOutput
    expected: ExpectedResolution
    metric_values: dict[str, float]  # metric_name -> value
    metric_details: dict[str, dict[str, object]] = Field(default_factory=dict)
