"""
common.py - Shared models, helpers, and sample data for the Chapter 9 labs
(customer-care edition).

Finance (insurance) edition -> customer-care edition:
  insurance claim (auto collision)   -> WARRANTY CLAIM on a NovaMart Care+ extended warranty
  insurance policy                   -> warranty PLAN (period, coverage limit, service fee, exclusions)
  First Notice of Loss (FNOL)        -> claim INTAKE
  police report                      -> TECHNICIAN JOB SHEET
  fraud screening / fraud debate     -> RETURN-ABUSE screening / abuse investigation debate
  deductible                         -> SERVICE FEE (visit charge the customer pays)
  underwriting a new policy          -> PRICING a Care+ extended warranty for a new purchase

The three labs share the same data contracts (WarrantyClaimRecord, WarrantyPlanRecord,
AbuseArgument, ...) and the same sample claim, so each notebook can focus on the
*architecture* it demonstrates.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_claim_id() -> str:
    short = uuid.uuid4().hex[:8].upper()
    return f"WCL-{date.today().year}-{short}"


def log_audit(state: dict, agent_name: str, action: str, detail: str = "") -> None:
    if "claim" not in state:
        return
    claim = state["claim"]
    claim.setdefault("audit_log", []).append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "action": action,
        "detail": detail[:500],
    })


def days_between(d1: str, d2: str) -> int:
    """Signed days from d1 to d2 (positive when d2 is later).

    Accepts ISO dates or datetimes; only the date part is compared.
    """
    fmt = "%Y-%m-%d"
    try:
        return (datetime.strptime(d2[:10], fmt)
                - datetime.strptime(d1[:10], fmt)).days
    except ValueError as exc:
        raise ValueError(
            f"Expected ISO dates (YYYY-MM-DD), got {d1!r}, {d2!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Sample data - used by Lab 1 (claims pipeline) and Lab 2 (abuse debate)
# ---------------------------------------------------------------------------

SAMPLE_CLAIM = {
    "name": "Ananya Iyer",
    "customer_id": "C1003",
    "plan_number": "CARE-2024-AQP-55102",
    "date_of_failure": "2026-10-22",
    "description": (
        "AquaPure RO water purifier stopped dispensing water and is leaking "
        "from the base. Red E2 error light is on. The unit has not been moved "
        "since installation."
    ),
}

SAMPLE_PLAN = {
    "plan_number": "CARE-2024-AQP-55102",
    "customer_name": "Ananya Iyer",
    "inception_date": "2025-11-25",      # Care+ starts when the 1-year maker warranty ends
    "expiration_date": "2027-11-25",
    "product_category": "kitchen",
    "coverage_limit": 18500,             # up to the invoice value of the product
    "service_fee": 199,                  # visit charge the customer pays per claim
    "coverage_sections": "NovaMart Care+ extended warranty (parts and labour)",
    "exclusions": ["physical damage", "unauthorised repair", "commercial use", "tampered seal"],
}

SAMPLE_DOCUMENTS = [
    {
        "type": "claim_form",
        "fields": {
            "customer": "Ananya Iyer",
            "date": "2026-10-22",
            "location": "Adyar, Chennai 600020",
            "product": "AquaPure RO Purifier AQP-500, serial AQP5-24-118842",
            "invoice": "ORD-71999 dated 2024-11-20",
            "description": "Purifier not dispensing water, leaking from the base",
        },
    },
    {
        "type": "photo",
        "description": (
            "Water pooled under the unit. Red E2 error light lit on the front "
            "panel. Outer casing intact, no cracks or impact marks."
        ),
    },
    {
        "type": "photo",
        "description": (
            "Close-up of the base: drip from the inlet solenoid valve connector. "
            "Warranty seal on the service panel intact."
        ),
    },
    {
        "type": "technician_report",
        "summary": (
            "Job sheet JS-2026-10-23-0912. Visit on 23 Oct 2026. Diagnosis: RO "
            "booster pump failed (no pressure); inlet solenoid valve leaking. No "
            "signs of impact, warranty seal intact, installed in a residence. "
            "Serial number matches the invoice. Recommended: replace pump and valve."
        ),
    },
]


# A suspicious-looking claim - used by Lab 2 to drive the abuse debate.
SUSPICIOUS_CLAIM_SUMMARY = (
    "Customer: Vikram Singh (C1004), Silver tier, member since June 2025. "
    "Order ORD-78411: Vistara 65in 4K TV, Rs 74,990, delivered 2026-10-20 and "
    "signed for without remarks. Claim raised 2026-10-24 (4 days later): screen "
    "cracked; customer wants a full refund, not a replacement, paid to a "
    "DIFFERENT card from the one used at checkout, and threatens a chargeback. "
    "Courier proof-of-delivery photo shows an intact box. No unboxing video. "
    "History: 3 'damaged in transit' claims in the last 6 months (soundbar, "
    "smartwatch, earbuds), 75% return rate over 12 months, negative margin to date. "
    "Technician visit: crack radiates from a single point of impact at the lower-left "
    "corner of the panel; panel serial number matches the invoice."
)
