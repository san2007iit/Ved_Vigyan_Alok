"""
orchestrator.py
---------------
The Orchestrator Agent: coordinates the whole case-resolution pipeline.

Architecture:
  The Orchestrator uses the five specialist agents as tools via the
  .as_tool() pattern of the OpenAI Agents SDK. The Orchestrator keeps
  control of the workflow and receives each agent's output before
  proceeding, unlike the handoff pattern where control is transferred.

Execution strategy:
  1. customer_profile_agent  -+
  2. voice_of_customer_agent -+-- run concurrently (independent data sources)
  3. churn_risk_agent        -+
  4. customer_value_agent     -- runs after the profile (needs spend and margin context)
  5. resolution_writer_agent  -- runs last (synthesises everything)
"""

from agents import Agent

from specialist_agents.churn_risk_agent import churn_risk_agent
from specialist_agents.config import MODEL
from specialist_agents.customer_profile_agent import customer_profile_agent
from specialist_agents.customer_value_agent import customer_value_agent
from specialist_agents.resolution_writer_agent import resolution_writer_agent
from specialist_agents.voice_of_customer_agent import voice_of_customer_agent

ORCHESTRATOR_INSTRUCTIONS = """
You are the escalation-desk lead orchestrating the analysis of a customer case.
You coordinate a team of specialist sub-agents and pass their findings to the writer.

Given a customer ID (and optionally the customer's latest message), execute this workflow:

PHASE 1 - DATA GATHERING (call all three simultaneously in a single step):

  Call customer_profile_analysis, voice_of_customer_analysis and churn_risk_assessment
  all at once with the customer ID. Do not wait for one to finish before calling the next.
  They are independent and must be issued as parallel tool calls in a single response.

  - customer_profile_analysis:  record, KPIs and a customer-health verdict.
  - voice_of_customer_analysis: what the customer says, sentiment, material risk flags.
  - churn_risk_assessment:      risk register and a churn-risk verdict.

PHASE 2 - CUSTOMER VALUE (requires the Phase 1 profile):

  Call customer_value_analysis with the following input (fill in the values from the
  customer_profile_analysis output). The customer ID must appear on the first line.

          Customer: [ID]

          ---CONTEXT FOR CLV---
          Tier:                                [Platinum | Gold | Silver]
          Tenure (years):                      [X]
          Spend growth 2025 vs 2024:           [X% or N/A]
          Trailing 12-month margin (INR):      [X]
          Margin % 2025:                       [X%]
          ---END CONTEXT---

          If a value is not available in the profile output, write "N/A".

PHASE 3 - SYNTHESIS:

  Call write_case_resolution with ALL findings from Phases 1 and 2. Structure your input as:
          ---
          CUSTOMER PROFILE ANALYSIS:
          [output from customer_profile_analysis]

          VOICE OF CUSTOMER ANALYSIS:
          [output from voice_of_customer_analysis]

          CHURN RISK ASSESSMENT:
          [output from churn_risk_assessment]

          CUSTOMER VALUE ANALYSIS:
          [output from customer_value_analysis]
          ---

IMPORTANT:
- Do not skip any step. All four specialist analyses are required.
- If a sub-agent returns an error, note it in the input to the writer rather than stopping.
- Your final output must be the ResolutionWriterAgent's output copied verbatim, character
  for character. Do not add commentary, remove sections, reformat or summarise it.
"""

orchestrator = Agent(
    name="OrchestratorAgent",
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    tools=[
        customer_profile_agent.as_tool(
            tool_name="customer_profile_analysis",
            tool_description=(
                "Reads the customer's profile, order history and ticket history and computes "
                "service and value KPIs. Returns a customer-health verdict."
            ),
        ),
        voice_of_customer_agent.as_tool(
            tool_name="voice_of_customer_analysis",
            tool_description=(
                "Retrieves the customer's recent messages, call notes, social posts and reviews "
                "and scores sentiment. Returns a sentiment verdict and material risk flags."
            ),
        ),
        churn_risk_agent.as_tool(
            tool_name="churn_risk_assessment",
            tool_description=(
                "Assesses relationship risks: churn, reputation, legal, safety and policy abuse. "
                "Returns a risk register and a churn-risk verdict."
            ),
        ),
        customer_value_agent.as_tool(
            tool_name="customer_value_analysis",
            tool_description=(
                "Estimates customer lifetime value with a CLV model and a tier-peer comparison. "
                "Returns CLV, a value verdict and a recovery-budget guideline."
            ),
        ),
        resolution_writer_agent.as_tool(
            tool_name="write_case_resolution",
            tool_description=(
                "Synthesises all analyses into the final Case Resolution Brief with a service "
                "tier, a resolution plan and a draft message to the customer."
            ),
        ),
    ],
    model=MODEL,
)
