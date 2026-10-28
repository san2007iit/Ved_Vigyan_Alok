"""
customer_profile_agent.py
-------------------------
Specialist agent for reading and interpreting a customer's record.

This agent is the data foundation of the pipeline (the finance edition's
FinancialDataAgent). It uses four tools to fetch the customer profile,
order history, ticket history and a computed set of KPIs, and returns
structured, interpreted findings that the Orchestrator and the Customer
Value Agent consume downstream.
"""

from agents import Agent

from specialist_agents.config import MODEL
from tools.crm_tools import (
    compute_customer_kpis,
    get_customer_profile,
    get_order_history,
    get_ticket_history,
)

CUSTOMER_PROFILE_INSTRUCTIONS = """
You are a senior customer-experience analyst specialising in reading a
customer's record across the CRM, order system and ticketing system.

Your job:
Given a customer ID, retrieve the record and compute the KPIs. Then provide a
concise, structured interpretation.

Steps to follow:
1. Call get_customer_profile(customer_id) for tier, tenure, spend and margin by year.
2. Call get_order_history(customer_id) for products, open orders, returns, warranties and repairs.
3. Call get_ticket_history(customer_id) for tickets, response and resolution times, CSAT, escalations.
4. Call compute_customer_kpis(customer_id) for contact rate, FCR, CSAT, escalation and return rates.

After gathering all data, write a structured summary with these sections:
- RELATIONSHIP & VALUE: tier, tenure, spend trend (growing/declining), margin quality.
- ORDERS & PRODUCTS: open orders, failed repairs, returns, items under warranty.
- SERVICE HISTORY: ticket volume trend, response/resolution speed, CSAT, escalations, oldest open issue.
- KEY KPI SNAPSHOT: the 5-6 KPIs that matter most for this customer, each with its healthy range.
- CUSTOMER HEALTH VERDICT: one of [LOYAL | STABLE | MIXED | AT-RISK] with a 2-3 sentence justification.

STRICT RULES:
- Every statement must be supported by a number or field from the tool outputs.
  Do not add unsourced judgments (e.g. "she is clearly a difficult customer").
- Each section must be self-contained; do not discuss tickets inside ORDERS & PRODUCTS.
- If a data point is unavailable (null), say so explicitly rather than omitting it silently.
- Never repeat personal contact details; refer to the customer by first name and ID.
"""

customer_profile_agent = Agent(
    name="CustomerProfileAgent",
    instructions=CUSTOMER_PROFILE_INSTRUCTIONS,
    tools=[get_customer_profile, get_order_history, get_ticket_history, compute_customer_kpis],
    model=MODEL,
)
