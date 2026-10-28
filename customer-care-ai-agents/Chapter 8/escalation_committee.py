"""
Multi-Agent Escalation Committee - implemented as a LangGraph state graph.
(Customer-care edition of the finance chapter's Investment Committee.)

Roles:
  - Four specialist analysts run in parallel:
      Customer Value   (finance: Fundamentals)  - CLV, tenure, spend, margin
      Service Trend    (finance: Technicals)    - contact spike, CSAT momentum, failed fixes
      Voice of Customer(finance: Sentiment)     - the customer's own messages
      Operations       (finance: Macro)         - backlog, spare parts, budget, competitors
  - The Case Manager (finance: Portfolio Manager) synthesises the four reports into a
    recommendation: COMPENSATE / STANDARD / DECLINE, with a goodwill credit size.
  - The Compliance Officer (finance: Risk Officer) applies hard rules and may veto.

LLMs (one instance per tier, reused across nodes):
  - Sonnet for the specialists (cost-efficient, strong reasoning on short inputs).
  - Haiku for the high-volume voice-of-customer reading.
  - Opus for the Case Manager and Compliance roles, where reasoning quality matters most.

Run:
    export ANTHROPIC_API_KEY=...
    python escalation_committee.py               # CASE-9031 (Priya, Platinum)
    python escalation_committee.py CASE-9044     # Vikram (return-abuse signals)
"""
from __future__ import annotations

import os
import re
import sys
from statistics import mean
from typing import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from cases import OPS_SNAPSHOT, get_case

load_dotenv()

# ---------------------------------------------------------------------------
# Models - one instance per tier, reused across nodes.
# ---------------------------------------------------------------------------
# We never set `temperature`: recent Claude models reject it, so behaviour is
# steered through the prompts instead.
# The beta header + fallbacks="default" (passed through model_kwargs) re-runs a
# request that a safety classifier declines on another Claude model. Haiku is
# called without it.
OPUS_MODEL = os.getenv("CC_OPUS_MODEL", "claude-opus-5-5")
SONNET_MODEL = os.getenv("CC_SONNET_MODEL", "claude-sonnet-5-5")
HAIKU_MODEL = os.getenv("CC_HAIKU_MODEL", "claude-haiku-5-5")


def claude(model: str, max_tokens: int = 16000, fallback: bool = True) -> ChatAnthropic:
    if not fallback:
        return ChatAnthropic(model=model, max_tokens=max_tokens)
    return ChatAnthropic(
        model=model,
        max_tokens=max_tokens,
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )


opus = claude(OPUS_MODEL)
sonnet = claude(SONNET_MODEL)
haiku = claude(HAIKU_MODEL, fallback=False)


def _text(message) -> str:
    """Return the text of a LangChain AI message. With thinking enabled the content
    is a list of blocks (thinking + text), so keep only the text blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


# ---------------------------------------------------------------------------
# Shared state - every node reads from and writes into this TypedDict.
# Each analyst writes a different field, so no reducer is needed for the
# parallel fan-out from START.
# ---------------------------------------------------------------------------
class CommitteeState(TypedDict):
    case_id: str
    value_report: str
    service_report: str
    voice_report: str
    operations_report: str
    manager_thesis: str
    compliance_verdict: str
    final_decision: str


# ---------------------------------------------------------------------------
# Data fetchers - small, deterministic Python functions that the analyst
# nodes call before invoking the LLM. Keeping the numbers in Python (not in
# the prompt) makes them auditable and avoids hallucinated figures.
#
# Unlike the finance edition (yfinance always returns TODAY's data), every
# fetcher reads from a point-in-time case snapshot, so the same committee can
# be replayed on historical cases in backtest.py without look-ahead.
# ---------------------------------------------------------------------------
def fetch_value_profile(case: dict) -> dict:
    return {
        "tier": case["tier"],
        "tenure_years": case["tenure_years"],
        "lifetime_spend_inr": case["lifetime_spend_inr"],
        "trailing_margin_inr": case["trailing_margin_inr"],
        "clv_estimate_inr": case["clv_estimate_inr"],
        "order_value_inr": case["order"]["value_inr"],
    }


def compute_service_indicators(case: dict) -> dict:
    s = case["service"]
    csat = s["csat_last_5"]
    recent, earlier = csat[-2:], csat[:-2] or csat[:1]
    baseline = max(s["avg_contacts_per_month_prior_90d"], 0.1)
    return {
        "contacts_last_30d": s["contacts_last_30d"],
        "contact_spike_x": round(s["contacts_last_30d"] / baseline, 1),
        "csat_recent_avg": round(mean(recent), 2),
        "csat_momentum": round(mean(recent) - mean(earlier), 2),
        "days_open": s["days_open"],
        "failed_fixes": s["failed_fixes"],
        "company_fault": case["company_fault"],
    }


def compute_abuse_risk(case: dict) -> float:
    """0-1 score from return behaviour; the compliance hard rule uses it."""
    a = case["abuse"]
    score = 0.5 * a["return_rate_12m"] + min(0.3, 0.1 * a["damaged_claims_6m"])
    if a["payment_risk_flag"]:
        score += 0.2
    return round(min(score, 1.0), 2)


def fetch_messages(case: dict, n: int = 8) -> list[str]:
    """Most recent customer messages, newest first, as '[date, channel] text'."""
    return case["messages"][:n]


def fetch_ops_snapshot(case: dict) -> dict:
    """Historical cases carry the snapshot from their decision date; live cases use today's."""
    return case.get("ops_snapshot", OPS_SNAPSHOT)


# ---------------------------------------------------------------------------
# Nodes - each is a pure function `state -> partial_state_update`.
# ---------------------------------------------------------------------------
def value_analyst(state: CommitteeState) -> dict:
    case = get_case(state["case_id"])
    data = fetch_value_profile(case)
    sys_msg = SystemMessage(content=(
        "You are a senior customer-value analyst. Given the value profile, write a SHORT "
        "report (max 150 words) covering: (1) how valuable this customer is to the business, "
        "(2) profitability (margin), (3) loyalty (tenure), (4) what is at stake if they leave. "
        "Conclude with a stance: FAVOR / NEUTRAL / AGAINST COMPENSATION on value grounds."
    ))
    msg = HumanMessage(content=f"Case: {state['case_id']}\nValue profile: {data}")
    return {"value_report": _text(sonnet.invoke([sys_msg, msg]))}


def service_analyst(state: CommitteeState) -> dict:
    case = get_case(state["case_id"])
    ind = compute_service_indicators(case)
    sys_msg = SystemMessage(content=(
        "You are a service-operations analyst. Given service indicators, write a SHORT "
        "report (max 120 words) on: the contact spike versus the customer's normal level, "
        "CSAT momentum (negative = getting worse), how long the case has been open, how many "
        "fixes failed, and whether NovaMart is at fault. Conclude with: "
        "FAVOR / NEUTRAL / AGAINST COMPENSATION on service grounds."
    ))
    msg = HumanMessage(content=f"Case: {state['case_id']}\nIssue: {case['issue']}\nIndicators: {ind}")
    return {"service_report": _text(sonnet.invoke([sys_msg, msg]))}


def voice_analyst(state: CommitteeState) -> dict:
    case = get_case(state["case_id"])
    messages = fetch_messages(case)
    body = "\n- ".join(messages) if messages else "(no messages available)"
    sys_msg = SystemMessage(content=(
        "You are a voice-of-customer analyst. Each line below is one customer message, "
        "newest first, formatted '[date, channel] text'. Use the metadata, not just the words:\n"
        "  - Weight RECENT messages more heavily than older ones (emotion changes).\n"
        "  - Treat PUBLIC channels (Twitter, app-store reviews) as higher brand risk than "
        "private ones (email, chat, phone).\n"
        "  - Flag explicit threats (consumer court, chargeback, switching to a competitor) "
        "and any safety words.\n"
        "Produce a SHORT report (max 120 words) and score sentiment as: "
        "POSITIVE / MIXED / NEGATIVE."
    ))
    msg = HumanMessage(content=f"Case: {state['case_id']}\nMessages:\n- {body}")
    return {"voice_report": _text(haiku.invoke([sys_msg, msg]))}


def operations_analyst(state: CommitteeState) -> dict:
    ops = fetch_ops_snapshot(get_case(state["case_id"]))
    sys_msg = SystemMessage(content=(
        "You are a customer-operations strategist. Given a short operations snapshot, write "
        "a SHORT report (max 120 words) on the current regime (stretched / normal / calm) "
        "and what it implies for spending goodwill budget on a single escalated case."
    ))
    msg = HumanMessage(content=f"Operations snapshot: {ops}")
    return {"operations_report": _text(sonnet.invoke([sys_msg, msg]))}


def case_manager(state: CommitteeState) -> dict:
    case = get_case(state["case_id"])
    sys_msg = SystemMessage(content=(
        "You are the Case Manager. Synthesise the four analyst reports into a single "
        "recommendation (max 250 words). The policy remedy (repair, replacement or refund "
        "the policy allows) happens in every case; you decide what goes ON TOP of it. "
        "End STRICTLY with one line of the form:\n"
        "ACTION=<COMPENSATE|STANDARD|DECLINE>; CONFIDENCE=<0-100>; CREDIT_PCT=<0.0-15.0>\n\n"
        "Where:\n"
        "  - ACTION    : COMPENSATE = policy remedy plus a goodwill credit;\n"
        "                STANDARD   = policy remedy only;\n"
        "                DECLINE    = refuse the claim and send it to investigation.\n"
        "  - CONFIDENCE: how strongly you back the call (0 = none, 100 = very high).\n"
        "  - CREDIT_PCT: the goodwill credit as a percentage of the ORDER VALUE.\n"
        "                Size purely on your conviction; do not anticipate any\n"
        "                downstream review. Use this scale:\n"
        "                  0.0      -> STANDARD or DECLINE (no goodwill).\n"
        "                  1.0-3.0  -> token gesture when the case is mixed.\n"
        "                  3.0-7.0  -> standard goodwill at moderate conviction (CONFIDENCE 50-70).\n"
        "                  7.0-10.0 -> strong goodwill at high conviction (CONFIDENCE 70+).\n"
        "                  10.1-15.0-> exceptional recovery. Reserve for cases where every\n"
        "                              analyst favours compensation AND NovaMart is clearly at fault.\n"
        "                COMPENSATE at CREDIT_PCT=0 is incoherent; pick STANDARD instead.\n"
        "                STANDARD or DECLINE must have CREDIT_PCT=0."
    ))
    msg = HumanMessage(content=(
        f"Case: {state['case_id']} - {case['customer']}\n"
        f"Order: {case['order']}\nIssue: {case['issue']}\n\n"
        f"--- Customer value ---\n{state['value_report']}\n\n"
        f"--- Service trend ---\n{state['service_report']}\n\n"
        f"--- Voice of customer ---\n{state['voice_report']}\n\n"
        f"--- Operations ---\n{state['operations_report']}"
    ))
    return {"manager_thesis": _text(opus.invoke([sys_msg, msg]))}


def _extract_action(thesis: str) -> str:
    upper = thesis.upper().replace(" ", "")
    if "ACTION=COMPENSATE" in upper:
        return "COMPENSATE"
    if "ACTION=DECLINE" in upper:
        return "DECLINE"
    return "STANDARD"


def compliance_officer(state: CommitteeState) -> dict:
    case = get_case(state["case_id"])
    abuse = compute_abuse_risk(case)

    # Hard, deterministic rule - checked before any LLM call.
    if abuse > 0.6:
        return {
            "compliance_verdict": f"VETO: abuse risk score {abuse:.2f} exceeds the 0.60 ceiling.",
            "final_decision": "STANDARD (compliance veto)",
        }

    sys_msg = SystemMessage(content=(
        "You are the Head of Customer-Care Compliance. Review the Case Manager's "
        "recommendation and reject it if any of the following are true: "
        "(a) COMPENSATE is recommended but the thesis ignores abuse signals or the precedent "
        "it sets for similar cases, "
        "(b) CREDIT_PCT > 10.0 (the single-case cap without CX-manager sign-off, policy KB-103), "
        "(c) CONFIDENCE below 50. Otherwise approve. Reply STRICTLY in the form:\n"
        "VERDICT=<APPROVED|REJECTED>; REASON=<short explanation>"
    ))
    msg = HumanMessage(content=(
        f"Case Manager thesis:\n{state['manager_thesis']}\n\n"
        f"Order value: Rs {case['order']['value_inr']:,}\nAbuse risk score: {abuse:.2f}"
    ))
    verdict_text = _text(opus.invoke([sys_msg, msg]))
    if "REJECTED" in verdict_text.upper():
        final = "STANDARD (compliance rejected)"
    else:
        final = _extract_action(state["manager_thesis"])
    return {"compliance_verdict": verdict_text, "final_decision": final}


# ---------------------------------------------------------------------------
# Graph assembly - START fans out to four analysts, all four converge on the
# Case Manager, the Case Manager hands off to Compliance, and we are done.
# ---------------------------------------------------------------------------
def build_committee():
    g = StateGraph(CommitteeState)
    g.add_node("customer_value", value_analyst)
    g.add_node("service_trend", service_analyst)
    g.add_node("voice_of_customer", voice_analyst)
    g.add_node("operations", operations_analyst)
    g.add_node("case_manager", case_manager)
    g.add_node("compliance_officer", compliance_officer)

    for analyst in ("customer_value", "service_trend", "voice_of_customer", "operations"):
        g.add_edge(START, analyst)
        g.add_edge(analyst, "case_manager")
    g.add_edge("case_manager", "compliance_officer")
    g.add_edge("compliance_officer", END)
    return g.compile()


def run_committee(case_id: str) -> dict:
    initial: CommitteeState = {
        "case_id": case_id,
        "value_report": "",
        "service_report": "",
        "voice_report": "",
        "operations_report": "",
        "manager_thesis": "",
        "compliance_verdict": "",
        "final_decision": "",
    }
    return build_committee().invoke(initial)


if __name__ == "__main__":
    case_id = sys.argv[1] if len(sys.argv) > 1 else "CASE-9031"
    result = run_committee(case_id)
    print("=" * 80)
    print(f"FINAL DECISION for {result['case_id']}: {result['final_decision']}")
    print("=" * 80)
    print("\n--- Case Manager thesis ---")
    print(result["manager_thesis"])
    print("\n--- Compliance verdict ---")
    print(result["compliance_verdict"])
    print("\n\n" + "=" * 80)
    print("FOUR ANALYST REPORTS")
    print("=" * 80)
    print("\n--- Customer value ---")
    print(result["value_report"])
    print("\n--- Service trend ---")
    print(result["service_report"])
    print("\n--- Voice of customer ---")
    print(result["voice_report"])
    print("\n--- Operations ---")
    print(result["operations_report"])
