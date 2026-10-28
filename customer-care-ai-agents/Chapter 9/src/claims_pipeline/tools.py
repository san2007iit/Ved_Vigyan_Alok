"""
tools.py
========
Tool functions for the warranty claims pipeline agents.

Each agent has a dedicated tool that reads from and writes to the
shared Context object - the shared-state communication pattern
from Chapter 7, Section 4.1. Every write also appends to the audit
trail, so any decision can be reconstructed later.
"""

import json

from llama_index.core.workflow import Context

from ..common import generate_claim_id, log_audit, days_between
from ..models import ProductCategory


def classify_product(description: str) -> ProductCategory:
    """Classify the product category from the description (simplified)."""
    d = description.lower()
    if any(w in d for w in ["refrigerator", "fridge", "air conditioner", " ac ",
                            "washing machine", "washer"]):
        return ProductCategory.LARGE_APPLIANCE
    if any(w in d for w in ["tv", "television", "phone", "laptop", "screen"]):
        return ProductCategory.ELECTRONICS
    if any(w in d for w in ["purifier", "mixer", "grinder", "microwave", "oven"]):
        return ProductCategory.KITCHEN
    return ProductCategory.ACCESSORY


async def process_intake(ctx: Context, claim_data: dict) -> str:
    """Create the warranty-claim record from the customer's request.

    Used by: IntakeAgent (root agent).
    Writes: claim record to shared state.
    """
    if isinstance(claim_data, str):  # some models pass the JSON as a string
        claim_data = json.loads(claim_data)
    claim_id = generate_claim_id()
    category = classify_product(claim_data.get("description", ""))

    claim = {
        "claim_id": claim_id,
        "customer_name": claim_data.get("name", ""),
        "plan_number": claim_data.get("plan_number", ""),
        "product_category": category.value,
        "date_of_failure": claim_data.get("date_of_failure", ""),
        "description": claim_data.get("description", ""),
        "status": "received",
        "parsed_documents": [],
        "audit_log": [],
    }

    state = await ctx.store.get("state")
    state["claim"] = claim
    log_audit(state, "IntakeAgent", "Claim created",
              f"ID: {claim_id}, Category: {category.value}")
    await ctx.store.set("state", state)
    return f"Intake processed. Claim {claim_id}. Category: {category.value}."


async def parse_evidence(ctx: Context) -> str:
    """Parse submitted evidence (claim form, photos, technician job sheet).

    Used by: EvidenceParser.
    Reads: submitted_documents from state.
    Writes: parsed_documents, fault_description to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]
    documents = state.get("submitted_documents", [])

    parsed, fault_parts = [], []
    for doc in documents:
        doc_type = doc.get("type", "unknown")
        if doc_type == "photo":
            # In production: call a vision model on the photo
            desc = doc.get("description", "Fault visible in photo.")
            parsed.append({"type": "fault_photo", "fault_description": desc})
            fault_parts.append(desc)
        elif doc_type == "claim_form":
            parsed.append({"type": "claim_form", "fields": doc.get("fields", {})})
        elif doc_type == "technician_report":
            summary = doc.get("summary", "")
            parsed.append({"type": "technician_report", "summary": summary})
            fault_parts.append(summary)

    claim["parsed_documents"] = parsed
    claim["fault_description"] = " ".join(fault_parts)
    claim["status"] = "parsing"
    log_audit(state, "EvidenceParser",
              f"Parsed {len(parsed)} documents",
              f"Types: {[d['type'] for d in parsed]}")
    await ctx.store.set("state", state)
    return f"Parsed {len(parsed)} documents."


def coverage_evidence(claim: dict, plan: dict) -> tuple[list[str], bool]:
    """Return supported denial codes and whether basic lab evidence is complete.

    Exclusion matching is a simplified lab rule, not a legal coverage assessment.
    """
    from datetime import date

    try:
        failure = date.fromisoformat(claim["date_of_failure"])
        start = date.fromisoformat(plan["inception_date"])
        end = date.fromisoformat(plan["expiration_date"])
    except (KeyError, TypeError, ValueError):
        return [], False
    if start > end:
        return [], False
    codes = []
    if not start <= failure <= end:
        codes.append("OUTSIDE_PLAN_PERIOD")
    exclusions = plan.get("exclusions")
    if not isinstance(exclusions, list) or not all(isinstance(x, str) and x.strip() for x in exclusions):
        return codes, False
    description = claim.get("description", "")
    if not isinstance(description, str) or not description.strip():
        return codes, False
    if any(x.lower() in description.lower() for x in exclusions):
        codes.append("PLAN_EXCLUSION")
    return codes, True


async def verify_coverage(ctx: Context) -> str:
    """Check whether the warranty plan covers this failure.

    Used by: CoverageAgent.
    Reads: claim, plan from state.
    Writes: coverage fields to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]
    plan = state.get("plan", {})

    reason_codes, evidence_complete = coverage_evidence(claim, plan)
    is_covered = evidence_complete and not reason_codes
    exclusions, reasoning_parts = [], []
    if not evidence_complete:
        reasoning_parts.append("Coverage evidence is incomplete or invalid; human review required.")

    if "OUTSIDE_PLAN_PERIOD" in reason_codes:
        reasoning_parts.append(
            f"Failure date {claim.get('date_of_failure', '')} outside plan period "
            f"({plan.get('inception_date', '')} to {plan.get('expiration_date', '')})."
        )

    plan_exclusions = plan.get("exclusions", [])
    description = claim.get("description", "")
    for exclusion in (plan_exclusions if isinstance(plan_exclusions, list) else []):
        if not isinstance(exclusion, str) or not exclusion.strip():
            continue
        if isinstance(description, str) and exclusion.lower() in description.lower():
            exclusions.append(exclusion)
    if exclusions:
        reasoning_parts.append(f"Exclusions found: {exclusions}")

    if is_covered and not exclusions:
        reasoning_parts.append(
            f"Failure covered under {plan.get('coverage_sections', 'the warranty plan')}. "
            f"Limit: Rs {plan.get('coverage_limit', 0):,.0f}. "
            f"Service fee: Rs {plan.get('service_fee', 0):,.0f}."
        )

    claim["coverage_verified"] = is_covered and not exclusions
    claim["applicable_limit"] = plan.get("coverage_limit", 0)
    claim["service_fee"] = plan.get("service_fee", 0)
    claim["exclusions_found"] = exclusions
    claim["coverage_reasoning"] = " ".join(reasoning_parts)
    claim["coverage_reason_codes"] = reason_codes
    claim["coverage_evidence_complete"] = evidence_complete
    claim["status"] = "coverage_check"

    log_audit(state, "CoverageAgent",
              f"Coverage: {'verified' if claim['coverage_verified'] else 'denied'}",
              claim["coverage_reasoning"])
    await ctx.store.set("state", state)
    return f"Coverage {'verified' if claim['coverage_verified'] else 'not verified'}."


async def screen_for_abuse(ctx: Context) -> str:
    """Screen the claim against warranty-abuse indicators.

    Used by: AbuseScreener.
    Reads: claim, plan, claims_history from state.
    Writes: abuse_score, red_flags, abuse_recommendation to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]
    plan = state.get("plan", {})
    red_flags, score = [], 0

    # Red flag 1: failure soon after the plan started (cover bought for a known fault?)
    failure = claim.get("date_of_failure", "")
    inception = plan.get("inception_date", "")
    if failure and inception:
        days = days_between(inception, failure)
        if days < 0:
            red_flags.append(f"Failure predates plan start by {-days} days")
            score += 25
        elif days < 30:
            red_flags.append(f"Failure {days} days after plan start")
            score += 25

    # Red flag 2: many prior warranty or 'damaged' claims
    prior_claims = state.get("claims_history", [])
    if len(prior_claims) >= 3:
        red_flags.append(f"{len(prior_claims)} prior claims in 12 months")
        score += 20

    # Red flag 3: late reporting (the cost estimate does not exist yet, so
    # amount-versus-limit checks belong after assessment, not here)
    if failure:
        delay = days_between(failure, state.get("intake_date", failure))
        if delay < 0:
            red_flags.append(f"Claim reported {-delay} days before the recorded failure date")
            score += 15
        elif delay > 30:
            red_flags.append(f"Claim reported {delay} days after the failure")
            score += 15

    score = min(score, 100)
    recommendation = "investigate" if score > 40 else "proceed"
    claim["abuse_score"] = score
    claim["red_flags"] = red_flags
    claim["abuse_recommendation"] = recommendation
    claim["status"] = "abuse_screening"

    log_audit(state, "AbuseScreener",
              f"Score: {score}/100. Recommendation: {recommendation}",
              f"Red flags: {red_flags}")
    await ctx.store.set("state", state)
    return f"Abuse screening complete. Score: {score}/100."


async def assess_repair(ctx: Context) -> str:
    """Estimate the covered repair cost from the diagnosis.

    Used by: AssessmentAgent.
    Reads: fault_description, service_fee, applicable_limit from claim.
    Writes: covered_amount, assessment_breakdown to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]

    # In production: parts catalogue + labour rates; for the lab, keyword rules
    fault = claim.get("fault_description", "").lower()
    fee = claim.get("service_fee", 199)
    limit = claim.get("applicable_limit", 18500)

    estimate = 1500  # default: minor part plus labour
    if "pump" in fault:
        estimate = 2800
    if "compressor" in fault:
        estimate = 9500
    if "solenoid" in fault or "valve" in fault:
        estimate += 650
    if "control board" in fault or "pcb" in fault:
        estimate += 3500
    if "beyond repair" in fault:
        estimate = limit * 0.8

    covered = max(0, min(estimate - fee, limit))
    remedy = "replacement" if estimate >= 0.7 * limit else "repair"
    breakdown = (
        f"Repair estimate: Rs {estimate:,.0f}. "
        f"Service fee paid by customer: Rs {fee:,.0f}. "
        f"Plan limit: Rs {limit:,.0f}. "
        f"Covered by NovaMart: Rs {covered:,.0f}. Remedy: {remedy}."
    )

    claim["covered_amount"] = covered
    claim["assessment_breakdown"] = breakdown
    claim["status"] = "assessment"
    log_audit(state, "AssessmentAgent", f"Covered amount: Rs {covered:,.0f}", breakdown)
    await ctx.store.set("state", state)
    return f"Assessment complete. Covered amount: Rs {covered:,.0f} ({remedy})."


async def make_decision(ctx: Context) -> str:
    """Make the claim decision with its reasoning chain.

    Used by: DecisionAgent.
    Reads: coverage, abuse, assessment data from claim.
    Writes: decision, decision_reasoning to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]

    evidence_codes, evidence_complete = coverage_evidence(claim, state.get("plan", {}))
    if not claim.get("coverage_verified", False):
        if evidence_complete and evidence_codes:
            decision = "DENIED"
            reason_codes = evidence_codes
            reasoning = (
                f"Claim denied on recorded coverage evidence: {evidence_codes}. "
                f"{claim.get('coverage_reasoning', '')}"
            )
        else:
            decision = "ESCALATED"
            reason_codes = ["INSUFFICIENT_COVERAGE_EVIDENCE"]
            reasoning = "Coverage could not be established from the supplied evidence; refer for human review."
    elif claim.get("abuse_recommendation") == "investigate":
        decision = "ESCALATED"
        reason_codes = ["ABUSE_REVIEW_REQUIRED"]
        reasoning = (
            f"Claim escalated for abuse investigation. "
            f"Abuse score: {claim.get('abuse_score', 0)}/100. "
            f"Red flags: {claim.get('red_flags', [])}."
        )
    else:
        decision = "APPROVED"
        reason_codes = ["COVERAGE_VERIFIED"]
        reasoning = (
            f"Claim approved. {claim.get('coverage_reasoning', 'Covered by the plan.')} "
            f"Abuse score: {claim.get('abuse_score', 0)}/100 (low risk). "
            f"Covered amount: Rs {claim.get('covered_amount', 0):,.0f}."
        )

    claim["decision"] = decision
    claim["decision_reasoning"] = reasoning
    claim["decision_reason_codes"] = reason_codes
    claim["status"] = decision.lower()
    log_audit(state, "DecisionAgent", f"Decision: {decision}", reasoning)
    await ctx.store.set("state", state)
    return f"Decision: {decision}."


async def validate_compliance(ctx: Context) -> str:
    """Apply the lab's consumer-protection evidence checks (not a legal certification).

    Used by: ComplianceAgent (guardrail - has veto power).
    Reads: decision, reasoning from claim.
    Writes: status update, compliance issues if found.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]
    issues = []

    # Check 1: a denial must cite reason codes the evidence supports
    if claim.get("decision") == "DENIED":
        evidence_codes, complete = coverage_evidence(claim, state.get("plan", {}))
        codes = claim.get("decision_reason_codes", [])
        if (not complete or not evidence_codes or not isinstance(codes, list)
                or not all(isinstance(code, str) for code in codes)
                or set(codes) != set(evidence_codes)):
            issues.append("Denial requires reason codes supported by the plan and the evidence.")
        if claim.get("coverage_verified") is not False:
            issues.append("Denial contradicts the recorded coverage result.")

    if claim.get("decision") not in {"APPROVED", "DENIED", "ESCALATED"}:
        issues.append("Unrecognized decision value.")
    if claim.get("decision") == "APPROVED":
        codes, complete = coverage_evidence(claim, state.get("plan", {}))
        if not complete or codes or claim.get("coverage_verified") is not True:
            issues.append("Approval requires complete, consistent coverage evidence.")
        if claim.get("abuse_recommendation") == "investigate":
            issues.append("Approval cannot bypass an abuse referral.")

    # Check 2: the customer is owed an explanation they can understand
    if len(claim.get("decision_reasoning", "")) < 50:
        issues.append("Decision reasoning is missing or too short to explain to the customer.")

    # Check 3: every pipeline stage ran
    for field in ["coverage_verified", "abuse_score", "covered_amount"]:
        if claim.get(field) is None:
            issues.append(f"Missing required field: {field}")

    if issues:
        claim["status"] = "compliance_review"
        claim["compliance_issues"] = issues
        log_audit(state, "ComplianceAgent", "BLOCKED - compliance issues", str(issues))
        await ctx.store.set("state", state)
        return f"BLOCKED: {issues}"
    claim["compliance_issues"] = []
    log_audit(state, "ComplianceAgent", "PASSED - lab evidence checks")
    await ctx.store.set("state", state)
    return "Decision validated against the lab's evidence checks."
