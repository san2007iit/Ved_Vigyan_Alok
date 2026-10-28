"""
models.py
=========
Pydantic data models for warranty claims, warranty plans, and decisions.

These schemas enforce contracts between agents - each agent knows
exactly what it receives and what it must produce.
"""

from datetime import date
from enum import Enum
from pydantic import BaseModel, Field


class ProductCategory(str, Enum):
    LARGE_APPLIANCE = "large_appliance"   # refrigerator, AC, washing machine
    ELECTRONICS = "electronics"           # TV, phone, laptop
    KITCHEN = "kitchen"                   # water purifier, mixer grinder, microwave
    ACCESSORY = "accessory"


class ClaimStatus(str, Enum):
    RECEIVED = "received"
    PARSING = "parsing"
    COVERAGE_CHECK = "coverage_check"
    ABUSE_SCREENING = "abuse_screening"
    ASSESSMENT = "assessment"
    DECISION = "decision"
    APPROVED = "approved"
    DENIED = "denied"
    ESCALATED = "escalated"
    COMPLIANCE_REVIEW = "compliance_review"


class AbuseDetermination(str, Enum):
    STRONG_ABUSE_INDICATORS = "strong_abuse_indicators"
    LIKELY_ABUSE = "likely_abuse"
    INCONCLUSIVE = "inconclusive"
    LIKELY_LEGITIMATE = "likely_legitimate"


# ---------------------------------------------------------------------------
# Pydantic models - typed contracts between agents
# ---------------------------------------------------------------------------

class WarrantyClaimRecord(BaseModel):
    """Core claim data - the shared state object for the claims pipeline."""
    claim_id: str = ""
    customer_name: str = ""
    plan_number: str = ""
    product_category: ProductCategory = ProductCategory.KITCHEN
    date_of_failure: date = Field(default_factory=date.today)
    description: str = ""
    status: ClaimStatus = ClaimStatus.RECEIVED

    parsed_documents: list[dict] = []
    fault_description: str = ""

    coverage_verified: bool | None = None
    applicable_limit: float | None = None
    service_fee: float | None = None
    exclusions_found: list[str] = []
    coverage_reasoning: str = ""
    coverage_reason_codes: list[str] = Field(default_factory=list)
    coverage_evidence_complete: bool = False

    abuse_score: int | None = None
    red_flags: list[str] = []
    abuse_recommendation: str = ""

    covered_amount: float | None = None
    assessment_breakdown: str = ""

    decision: str = ""
    decision_reasoning: str = ""
    decision_reason_codes: list[str] = Field(default_factory=list)
    compliance_issues: list[str] = Field(default_factory=list)

    audit_log: list[dict] = []


class WarrantyPlanRecord(BaseModel):
    plan_number: str
    customer_name: str
    inception_date: date
    expiration_date: date
    product_category: ProductCategory
    coverage_limit: float
    service_fee: float
    coverage_sections: str = ""
    exclusions: list[str] = []


class AbuseArgument(BaseModel):
    """Structured argument for the abuse-investigation debate."""
    position: str  # "abuse" or "legitimate"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = []
    reasoning: str = ""
    weaknesses: list[str] = []


class ClaimDecision(BaseModel):
    decision: str  # "APPROVED" / "DENIED" / "ESCALATED"
    covered_amount: float = 0.0
    reasoning: str = ""
    policy_rules_applied: list[str] = []
