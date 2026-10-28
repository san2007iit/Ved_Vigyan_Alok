"""Reference-based metrics: structured outputs compared against gold labels.

Two metrics live here:
- decision_accuracy — exact match on the four-class decision
  (APPROVE / PARTIAL / ESCALATE / DENY)
- safety_escalation_recall — recall on "is this a product-safety hazard
  that must go to the safety team?"

Both report a Wilson 95% confidence interval alongside the point estimate,
because a committee needs the uncertainty around a metric, not just the
point value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..models import ComplaintCase, EvaluationResult, ResolutionAgentOutput


def decision_accuracy(results: Iterable[EvaluationResult]) -> tuple[float, tuple[float, float], int]:
    items = list(results)
    correct = sum(1 for r in items if r.output.decision == r.expected.decision)
    n = len(items)
    if n == 0:
        return 0.0, (0.0, 0.0), 0
    p = correct / n
    return p, _wilson_ci(correct, n), n


def safety_escalation_recall(results: Iterable[EvaluationResult]) -> tuple[float, tuple[float, float], int]:
    """Recall on confirmed safety hazards.

    Note: this is RECALL, not accuracy. Missing a real hazard (a smoking AC
    handled as a routine repair) is a catastrophic failure; escalating a
    harmless "phone gets warm" complaint costs specialist time and is
    tracked separately as the false-positive rate.
    """
    confirmed = [r for r in results if r.expected.safety_escalation]
    if not confirmed:
        return 1.0, (1.0, 1.0), 0
    detected = sum(1 for r in confirmed if r.output.safety_escalation)
    return detected / len(confirmed), _wilson_ci(detected, len(confirmed)), len(confirmed)


def safety_false_positive_rate(results: Iterable[EvaluationResult]) -> float:
    negatives = [r for r in results if not r.expected.safety_escalation]
    if not negatives:
        return 0.0
    return sum(1 for r in negatives if r.output.safety_escalation) / len(negatives)


def score_case(case: ComplaintCase, output: ResolutionAgentOutput) -> dict[str, float]:
    """Per-case reference-based scores. Aggregated upstream."""
    return {
        "decision_correct": float(output.decision == case.expected.decision),
        "safety_flag_correct": float(output.safety_escalation == case.expected.safety_escalation),
    }


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion.

    Preferred over the normal-approximation interval at small n or extreme p.
    """
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
