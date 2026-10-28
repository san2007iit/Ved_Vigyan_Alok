"""
Backtesting the Escalation Committee - historical case replay.
(Customer-care edition of the finance chapter's walk-forward backtest.)

We replay the committee on closed cases from Q2-Q3 2026, in the order they
were decided, and score each decision with NovaMart's uplift model:

    value saved  = (churn_if_standard - churn_if_compensated) x effectiveness x CLV
    net value    = value saved - goodwill credit paid
    effectiveness= min(1, CREDIT_PCT / 5)   # a credit of 5% of the order value or more
                                            # achieves the full uplift seen in past A/B
                                            # tests; a token gesture achieves part of it

Finance -> customer care:
    trade P&L            -> net value of a goodwill decision (CLV saved - credit paid)
    hit rate             -> share of compensations that created value
    equity curve         -> cumulative net value, case by case
    max drawdown         -> largest fall in cumulative net value
    Sharpe ratio         -> ROI on goodwill spend (net value / credits paid)

What this replay does better than the finance edition:
  - Every input is POINT-IN-TIME. Each historical case is a snapshot of what was
    known on its decision date (including the operations picture), and the
    churn estimates used for scoring are never shown to the committee.

What it does NOT do, and why:
  - The churn numbers are estimates from an uplift model, not observed outcomes:
    we never see what would have happened under the other decision. The real
    test is a randomised hold-out (some eligible cases get standard handling).
  - It ignores second-order effects: word of mouth from a public complaint, the
    precedent a generous credit sets, the cost of agent time.

Run:
    export ANTHROPIC_API_KEY=...
    python backtest.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from cases import HISTORICAL_CASES
from escalation_committee import run_committee

FULL_EFFECT_CREDIT_PCT = 5.0   # credit (as % of order value) that achieves the full uplift


# ---------------------------------------------------------------------------
# Decision record - one entry per replayed case.
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    case_id: str
    decided_on: str
    action: str             # COMPENSATE, STANDARD or DECLINE (after compliance)
    credit_pct: float       # goodwill credit as % of order value
    confidence: int
    order_value_inr: float
    clv_inr: float
    churn_if_standard: float
    churn_if_compensated: float

    @property
    def credit_inr(self) -> float:
        return self.order_value_inr * self.credit_pct / 100.0

    @property
    def clv_saved_inr(self) -> float:
        if self.action != "COMPENSATE":
            return 0.0
        effectiveness = min(1.0, self.credit_pct / FULL_EFFECT_CREDIT_PCT)
        uplift = self.churn_if_standard - self.churn_if_compensated
        return uplift * effectiveness * self.clv_inr

    @property
    def net_value_inr(self) -> float:
        return self.clv_saved_inr - self.credit_inr

    @property
    def oracle_action(self) -> str:
        """What a perfectly informed manager would do: compensate at the full-effect
        credit only when the expected CLV saved exceeds the credit paid."""
        uplift = self.churn_if_standard - self.churn_if_compensated
        best_credit = self.order_value_inr * FULL_EFFECT_CREDIT_PCT / 100.0
        return "COMPENSATE" if uplift * self.clv_inr > best_credit else "STANDARD"


# ---------------------------------------------------------------------------
# Decision parsing - pull ACTION, CONFIDENCE, CREDIT_PCT out of the Case
# Manager's trailer, honouring the Compliance Officer's veto if present.
# ---------------------------------------------------------------------------
_ACTION_RE = re.compile(r"ACTION\s*=\s*(COMPENSATE|STANDARD|DECLINE)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE\s*=\s*(\d+)", re.IGNORECASE)
_CREDIT_RE = re.compile(r"CREDIT_PCT\s*=\s*([\d.]+)", re.IGNORECASE)


def parse_decision(committee_result: dict) -> tuple[str, int, float]:
    thesis = committee_result["manager_thesis"]
    action_match = _ACTION_RE.search(thesis)
    action = action_match.group(1).upper() if action_match else "STANDARD"
    conf_match = _CONF_RE.search(thesis)
    credit_match = _CREDIT_RE.search(thesis)
    confidence = int(conf_match.group(1)) if conf_match else 0
    credit_pct = float(credit_match.group(1).rstrip(".")) if credit_match else 0.0

    final = committee_result["final_decision"].lower()
    if "(compliance veto)" in final or "(compliance rejected)" in final:
        return "STANDARD", confidence, 0.0
    if action != "COMPENSATE":
        credit_pct = 0.0
    return action, confidence, credit_pct


# ---------------------------------------------------------------------------
# Replay loop - cases in the order they were decided.
# ---------------------------------------------------------------------------
def replay(case_ids: list[str] | None = None) -> list[Decision]:
    ids = case_ids or sorted(HISTORICAL_CASES, key=lambda c: HISTORICAL_CASES[c]["decided_on"])
    decisions: list[Decision] = []
    for cid in ids:
        case = HISTORICAL_CASES[cid]
        result = run_committee(cid)
        action, confidence, credit_pct = parse_decision(result)
        decisions.append(Decision(
            case_id=cid, decided_on=case["decided_on"], action=action,
            credit_pct=credit_pct, confidence=confidence,
            order_value_inr=case["order"]["value_inr"], clv_inr=case["clv_estimate_inr"],
            churn_if_standard=case["churn_if_standard"],
            churn_if_compensated=case["churn_if_compensated"],
        ))
        print(f"  {cid} ({case['decided_on']}): {result['final_decision']:<32} "
              f"credit {credit_pct:4.1f}%  net Rs {decisions[-1].net_value_inr:>9,.0f}")
    return decisions


# ---------------------------------------------------------------------------
# Performance metrics.
# ---------------------------------------------------------------------------
def cumulative_curve(decisions: list[Decision]) -> list[float]:
    total, curve = 0.0, []
    for d in decisions:
        total += d.net_value_inr
        curve.append(total)
    return curve


def max_drawdown(curve: list[float]) -> float:
    """Largest fall (in Rs) from a running peak of cumulative net value."""
    peak, worst = 0.0, 0.0
    for value in curve:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def metrics(decisions: list[Decision]) -> dict:
    compensated = [d for d in decisions if d.action == "COMPENSATE"]
    credits = sum(d.credit_inr for d in decisions)
    net = sum(d.net_value_inr for d in decisions)
    curve = cumulative_curve(decisions)
    return {
        "cases": len(decisions),
        "compensated": len(compensated),
        "hit_rate": (sum(1 for d in compensated if d.net_value_inr > 0) / len(compensated)
                     if compensated else float("nan")),
        "credits_paid_inr": credits,
        "clv_saved_inr": sum(d.clv_saved_inr for d in decisions),
        "net_value_inr": net,
        "roi": net / credits if credits else float("nan"),
        "max_drawdown_inr": max_drawdown(curve),
        "missed_recoveries": [d.case_id for d in decisions
                              if d.oracle_action == "COMPENSATE" and d.action != "COMPENSATE"],
        "wasted_goodwill": [d.case_id for d in decisions
                            if d.action == "COMPENSATE" and d.oracle_action != "COMPENSATE"],
        "agreement_with_oracle": sum(1 for d in decisions
                                     if (d.action == "COMPENSATE") == (d.oracle_action == "COMPENSATE"))
                                 / len(decisions) if decisions else float("nan"),
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Replaying the committee on historical cases...")
    decisions = replay()
    m = metrics(decisions)

    print("=" * 80)
    print(f"Cases replayed          : {m['cases']}  ({decisions[0].decided_on} to {decisions[-1].decided_on})")
    print(f"Compensated             : {m['compensated']}")
    print(f"Hit rate                : {m['hit_rate']:.1%}  (compensations that created value)")
    print(f"Goodwill credits paid   : Rs {m['credits_paid_inr']:,.0f}")
    print(f"Expected CLV saved      : Rs {m['clv_saved_inr']:,.0f}")
    print(f"Net value               : Rs {m['net_value_inr']:,.0f}")
    print(f"ROI on goodwill         : {m['roi']:.1f}x")
    print(f"Max drawdown            : Rs {m['max_drawdown_inr']:,.0f}")
    print(f"Agreement with oracle   : {m['agreement_with_oracle']:.0%}")
    print(f"Missed recoveries       : {', '.join(m['missed_recoveries']) or 'none'}")
    print(f"Wasted goodwill         : {', '.join(m['wasted_goodwill']) or 'none'}")
    print("Cumulative net value    : " + "  ".join(f"{v:,.0f}" for v in m["curve"]))
    print("=" * 80)
