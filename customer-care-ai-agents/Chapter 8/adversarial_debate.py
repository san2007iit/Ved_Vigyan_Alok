"""
Adversarial Debate layer - implemented as a LangGraph state graph that runs
ON TOP of the Escalation Committee from `escalation_committee.py`.
(Customer-care edition of the finance chapter's Bull / Bear debate.)

The committee's recommendation becomes the seed for a structured debate:
  - A CUSTOMER ADVOCATE argues for generous recovery  (finance: Bull).
  - A COST CONTROLLER argues against spending goodwill (finance: Bear).
  - A Devil's Advocate identifies the weakest claim on each side.
  - A neutral Judge issues a final verdict that may differ from the committee.

Why heterogeneous models matter:
  We deliberately use TWO different model families (Anthropic + OpenAI) for
  the advocate and controller roles, because two instances of the same model
  produce correlated errors and the debate collapses into agreement.

Run:
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    python adversarial_debate.py                 # CASE-9031
    python adversarial_debate.py CASE-9050       # Rahul (late delivery, polite)
"""
from __future__ import annotations

import os
import sys
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from cases import get_case
from escalation_committee import OPUS_MODEL, SONNET_MODEL, _text, claude, run_committee

# ---------------------------------------------------------------------------
# Heterogeneous model line-up. The advocate and the controller are
# intentionally from different families so that they bring different priors
# to the debate; that family diversity, not a temperature setting, is what
# drives the disagreement.
# ---------------------------------------------------------------------------
advocate_model = claude(OPUS_MODEL)
controller_model = ChatOpenAI(model=os.getenv("CC_OPENAI_MODEL", "gpt-4o"), max_tokens=1500)
devil_model = claude(SONNET_MODEL)
judge_model = claude(SONNET_MODEL)


class DebateState(TypedDict):
    case_id: str
    case_summary: str
    committee_thesis: str
    committee_decision: str
    advocate_case: str
    controller_case: str
    devil_critique: str
    judge_verdict: str
    final_decision: str


def advocate_node(state: DebateState) -> dict:
    sys_msg = SystemMessage(content=(
        "You are the CUSTOMER ADVOCATE. Construct the strongest possible case for a GENEROUS "
        "recovery (goodwill credit on top of the policy remedy). Cite the strongest evidence: "
        "customer lifetime value, NovaMart's fault, failed fixes, public brand risk, competitor "
        "offers. Be concrete and do NOT hedge. Max 200 words. End with: ADVOCATE_CONVICTION=<0-100>."
    ))
    msg = HumanMessage(content=(
        f"Case: {state['case_summary']}\n"
        f"Committee thesis:\n{state['committee_thesis']}\n"
        f"Committee decision: {state['committee_decision']}"
    ))
    return {"advocate_case": _text(advocate_model.invoke([sys_msg, msg]))}


def controller_node(state: DebateState) -> dict:
    sys_msg = SystemMessage(content=(
        "You are the COST CONTROLLER. Construct the strongest possible case AGAINST spending "
        "goodwill budget on this case. Surface risks the committee may have under-weighted: "
        "precedent for similar customers, abuse signals, a budget that is already stretched, "
        "the possibility that the policy remedy alone resolves the case. Be concrete and do "
        "NOT hedge. Max 200 words. End with: CONTROLLER_CONVICTION=<0-100>."
    ))
    msg = HumanMessage(content=(
        f"Case: {state['case_summary']}\n"
        f"Committee thesis:\n{state['committee_thesis']}\n"
        f"Committee decision: {state['committee_decision']}"
    ))
    return {"controller_case": controller_model.invoke([sys_msg, msg]).content}


def devil_advocate(state: DebateState) -> dict:
    sys_msg = SystemMessage(content=(
        "You are a Devil's Advocate. Identify the SINGLE weakest claim in the CUSTOMER ADVOCATE "
        "case and the SINGLE weakest claim in the COST CONTROLLER case, and explain in one "
        "sentence each why those claims are weak. Max 150 words."
    ))
    msg = HumanMessage(content=(
        f"CUSTOMER ADVOCATE case:\n{state['advocate_case']}\n\n"
        f"COST CONTROLLER case:\n{state['controller_case']}"
    ))
    return {"devil_critique": _text(devil_model.invoke([sys_msg, msg]))}


def _parse_verdict(text: str) -> str:
    upper = text.upper().replace(" ", "")
    for action in ("COMPENSATE", "DECLINE", "STANDARD"):
        if f"VERDICT={action}" in upper:
            return action
    return "STANDARD"


def judge_node(state: DebateState) -> dict:
    sys_msg = SystemMessage(content=(
        "You are a NEUTRAL Judge. Weigh the customer advocate's case, the cost controller's "
        "case, and the devil's critique. Issue a verdict that may differ from the committee's "
        "prior decision. Reply STRICTLY in the form:\n"
        "VERDICT=<COMPENSATE|STANDARD|DECLINE>\nCONFIDENCE=<0-100>\nRATIONALE=<2-3 sentences>"
    ))
    msg = HumanMessage(content=(
        f"Committee prior: {state['committee_decision']}\n\n"
        f"CUSTOMER ADVOCATE case:\n{state['advocate_case']}\n\n"
        f"COST CONTROLLER case:\n{state['controller_case']}\n\n"
        f"Devil's critique:\n{state['devil_critique']}"
    ))
    verdict = _text(judge_model.invoke([sys_msg, msg]))
    return {"judge_verdict": verdict, "final_decision": _parse_verdict(verdict)}


def build_debate():
    g = StateGraph(DebateState)
    g.add_node("advocate", advocate_node)
    g.add_node("controller", controller_node)
    g.add_node("devil", devil_advocate)
    g.add_node("judge", judge_node)

    # Advocate and controller in parallel; both feed the devil; devil feeds the judge.
    g.add_edge(START, "advocate")
    g.add_edge(START, "controller")
    g.add_edge("advocate", "devil")
    g.add_edge("controller", "devil")
    g.add_edge("devil", "judge")
    g.add_edge("judge", END)
    return g.compile()


def run_pipeline(case_id: str) -> dict:
    """Run the full pipeline: committee first, then debate on the committee output."""
    committee_result = run_committee(case_id)
    case = get_case(case_id)
    initial: DebateState = {
        "case_id": case_id,
        "case_summary": f"{case_id} - {case['customer']}, {case['tier']}; "
                        f"{case['order']['product']} (Rs {case['order']['value_inr']:,}); {case['issue']}",
        "committee_thesis": committee_result["manager_thesis"],
        "committee_decision": committee_result["final_decision"],
        "advocate_case": "",
        "controller_case": "",
        "devil_critique": "",
        "judge_verdict": "",
        "final_decision": "",
    }
    debate_result = build_debate().invoke(initial)
    return {"committee": committee_result, "debate": debate_result}


if __name__ == "__main__":
    case_id = sys.argv[1] if len(sys.argv) > 1 else "CASE-9031"
    out = run_pipeline(case_id)
    print("=" * 80)
    print(f"COMMITTEE said     : {out['committee']['final_decision']}")
    print(f"DEBATE judge says  : {out['debate']['final_decision']}")
    print("=" * 80)
    print("\n--- Customer advocate ---\n", out["debate"]["advocate_case"])
    print("\n--- Cost controller ---\n", out["debate"]["controller_case"])
    print("\n--- Devil's critique ---\n", out["debate"]["devil_critique"])
    print("\n--- Judge verdict ---\n", out["debate"]["judge_verdict"])
