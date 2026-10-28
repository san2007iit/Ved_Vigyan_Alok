"""
agents.py
=========
Return-abuse investigation agents using the adversarial debate pattern.

The Abuse Advocate argues FOR deliberate abuse.
The Customer Advocate argues the claim is genuine.
The Investigation Decision Agent weighs both arguments.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from ..common import MODEL

llm = OpenAI(model=MODEL, temperature=0)


abuse_advocate = FunctionAgent(
    name="AbuseAdvocate",
    description="Argue the case for return or warranty abuse.",
    system_prompt=(
        "You are a returns-abuse investigator at an electronics retailer. Review the "
        "claim and build the strongest possible case that it is deliberate abuse "
        "(false damage claim, wardrobing, refund fraud). Cite specific evidence for each "
        "point. Also identify the weaknesses in your own argument: where is your case "
        "thin? Be thorough but honest. When done, hand off to CustomerAdvocate so the "
        "other side can respond."
    ),
    llm=llm, tools=[], can_handoff_to=["CustomerAdvocate"],
)

customer_advocate = FunctionAgent(
    name="CustomerAdvocate",
    description="Argue the claim is genuine.",
    system_prompt=(
        "You are a consumer-protection advocate. For every red flag the AbuseAdvocate "
        "raised, provide an innocent alternative explanation. Build the strongest "
        "possible case that the claim is genuine, and acknowledge your own weaknesses. "
        "Your role is to make sure no honest customer is wrongly refused or treated "
        "as a suspect. When done, hand off to InvestigationDecision."
    ),
    llm=llm, tools=[], can_handoff_to=["InvestigationDecision"],
)

investigation_decision = FunctionAgent(
    name="InvestigationDecision",
    description="Weigh the abuse and customer arguments.",
    system_prompt=(
        "You are the senior trust-and-safety investigator. You have arguments from an "
        "Abuse Advocate and a Customer Advocate. Evaluate:\n"
        "1. Evidence quality (documented vs. circumstantial)\n"
        "2. Reasoning strength (logical vs. speculative)\n"
        "3. Acknowledged weaknesses on each side\n\n"
        "Produce a referral signal:\n"
        "- strong_abuse_indicators: strong evidence of deliberate abuse\n"
        "- likely_abuse: the balance of evidence suggests abuse\n"
        "- inconclusive: insufficient evidence either way\n"
        "- likely_legitimate: the balance of evidence supports the customer\n\n"
        "Include a confidence score (0.0-1.0) and a recommended next action (for "
        "example: replacement after inspection, refund only to the original payment "
        "method, refer to the loss-prevention team). A person makes the final call."
    ),
    llm=llm, tools=[], can_handoff_to=[],
)
