"""
validator.py
============
Deterministic validation for the Deep Search Agent.

Validates research findings with code-based checks rather than LLM-based
verification: deterministic validation is faster, cheaper and more
reliable than asking an LLM "is this correct?".

Checks include:
- Completeness: all tasks completed, all product lines covered
- Numerical validity: metrics within plausible ranges
- Consistency: no contradictory data (e.g. first-contact resolution plus
  repeat-contact rate above 100%, or a contact rate that does not match
  tickets / orders)
"""

import math

from common import ProductLineMetrics, ResearchPlan, TaskStatus, ValidationResult


def validate_research(plan: ResearchPlan, results: dict[int, str],
                      required_product_lines: list[str]) -> ValidationResult:
    """Validate research findings using deterministic checks."""
    errors: list[str] = []
    warnings: list[str] = []
    gaps: list[str] = []

    # Check 1: task completion
    if not plan.sub_tasks:
        gaps.append("Research plan contains no tasks")
    for task in plan.sub_tasks:
        if task.status != TaskStatus.COMPLETED:
            gaps.append(f"Task {task.id} is not completed: {task.description}")
    completed = sum(1 for t in plan.sub_tasks if t.status == TaskStatus.COMPLETED)
    total = len(plan.sub_tasks)
    if total and completed < total:
        pct = completed / total * 100
        if pct < 50:
            errors.append(f"Only {completed}/{total} tasks completed ({pct:.0f}%) - insufficient data for synthesis")
        else:
            warnings.append(f"{completed}/{total} tasks completed ({pct:.0f}%)")

    # Check 2: only completed tasks with parsed metrics establish coverage
    found: set[str] = set()
    parsed: list[ProductLineMetrics] = []
    for task in plan.sub_tasks:
        if task.status != TaskStatus.COMPLETED:
            continue
        result = results.get(task.id)
        if not isinstance(result, str) or not result.strip():
            gaps.append(f"Task {task.id} has no result")
            continue
        try:
            metrics = ProductLineMetrics.model_validate_json(result)
        except (ValueError, TypeError):
            if "support_analytics" in task.data_sources and not task.dependencies:
                errors.append(f"Task {task.id} did not return valid support metrics")
            continue  # qualitative text cannot establish metrics coverage
        parsed.append(metrics)
        _validate_metrics(metrics, errors, warnings)
        found.add(metrics.product_line.upper())

    for code in sorted({p.upper() for p in required_product_lines} - found):
        gaps.append(f"Missing structured support metrics for required product line: {code}")

    # Check 3: cross-product consistency
    if len(parsed) >= 2:
        periods = {m.period for m in parsed}
        if len(periods) > 1:
            detail = ", ".join(f"{m.product_line}={m.period}" for m in parsed)
            warnings.append(f"Period mismatch across product lines: {detail}. Comparisons may not be like-for-like.")

    return ValidationResult(is_valid=not errors and not gaps, errors=errors, warnings=warnings, gaps=gaps)


def _validate_metrics(m: ProductLineMetrics, errors: list[str], warnings: list[str]) -> None:
    """Validate one ProductLineMetrics instance for plausibility and internal consistency."""
    code = m.product_line
    if not m.period or not m.period.strip():
        errors.append(f"{code}: missing reported period")
        return
    for field in ("contact_rate", "csat", "first_contact_resolution", "repeat_contact_rate",
                  "avg_resolution_hours"):
        value = getattr(m, field)
        if value is None or not math.isfinite(value):
            errors.append(f"{code}: {field} must be a finite number")
            return

    if m.orders <= 0:
        errors.append(f"{code}: orders must be positive (got {m.orders})")
        return
    if not 1 <= m.csat <= 5:
        errors.append(f"{code}: CSAT {m.csat} outside the 1-5 scale")
    for field in ("first_contact_resolution", "repeat_contact_rate", "return_rate"):
        value = getattr(m, field)
        if value is not None and not 0 <= value <= 100:
            errors.append(f"{code}: {field} {value}% outside 0-100%")

    # Consistency 1: the contact rate must match tickets / orders.
    implied = m.tickets / m.orders * 100
    if abs(implied - m.contact_rate) > 0.5:
        errors.append(f"{code}: contact rate {m.contact_rate} does not match tickets/orders = {implied:.1f}")

    # Consistency 2: a ticket cannot be both solved at first contact and need a repeat contact.
    if m.first_contact_resolution + m.repeat_contact_rate > 101:
        errors.append(f"{code}: first-contact resolution ({m.first_contact_resolution}%) + repeat-contact "
                      f"rate ({m.repeat_contact_rate}%) exceed 100% - mathematically impossible")

    # Plausibility warnings (not errors)
    if m.contact_rate > 50:
        warnings.append(f"{code}: contact rate {m.contact_rate} per 100 orders is extreme - verify ticket mapping")
    if m.csat_change_qoq is not None and abs(m.csat_change_qoq) >= 0.5:
        warnings.append(f"{code}: CSAT moved {m.csat_change_qoq:+.1f} points in one quarter - a material shift")


def format_validation(result: ValidationResult) -> str:
    lines = ["[PASS] Validation passed" if result.is_valid else "[FAIL] Validation failed"]
    if result.errors:
        lines.append("\n  Errors:")
        lines += [f"    [X] {e}" for e in result.errors]
    if result.warnings:
        lines.append("\n  Warnings:")
        lines += [f"    [!] {w}" for w in result.warnings]
    if result.gaps:
        lines.append("\n  Gaps:")
        lines += [f"    [ ] {g}" for g in result.gaps]
    return "\n".join(lines)
