"""
common.py
=========
Shared data models, configuration, and utility functions for the
Chapter 5 Deep Search Agent (customer-care edition).

All Pydantic models that flow between components are defined here,
establishing the contracts between Planner, Researcher, Validator,
and Synthesizer.
"""

import os
import re
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
_ = load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # optional: live web search; mock results otherwise

# Model configuration - PydanticAI model strings.
# Format: "provider:model-name" (e.g., "openai:gpt-4o", "anthropic:claude-opus-5-5")
PLANNING_MODEL = os.getenv("CC_PLANNING_MODEL", "openai:gpt-4o")
ROUTING_MODEL = os.getenv("CC_ROUTING_MODEL", "openai:gpt-4o-mini")
SYNTHESIS_MODEL = os.getenv("CC_SYNTHESIS_MODEL", "openai:gpt-4o")
SUMMARIZATION_MODEL = os.getenv("CC_SUMMARIZATION_MODEL", "openai:gpt-4o-mini")

# Research loop configuration
MAX_REPLAN_ATTEMPTS = 2
MAX_CONTEXT_TOKENS = 80_000  # soft limit for context management


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    """Status of a research sub-task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DataSource(str, Enum):
    """Available data sources for research tools."""
    SUPPORT_ANALYTICS = "support_analytics"   # finance edition: financial_api
    CX_DOCUMENTS = "cx_documents"             # finance edition: sec_filing
    WEB_SEARCH = "web_search"                 # same in both editions


# ---------------------------------------------------------------------------
# Research plan models
# ---------------------------------------------------------------------------

class SubTask(BaseModel):
    """A single research sub-task within a plan."""
    id: int
    description: str
    data_sources: list[str] = Field(
        description="Which tools to use: 'support_analytics', 'cx_documents', 'web_search'"
    )
    dependencies: list[int] = Field(
        default_factory=list,
        description="IDs of tasks that must complete before this one",
    )
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None


class ResearchPlan(BaseModel):
    """A structured research plan with sub-tasks and dependencies."""
    question: str
    sub_tasks: list[SubTask]


# ---------------------------------------------------------------------------
# Support-analytics data model (finance edition: CompanyMetrics)
# ---------------------------------------------------------------------------

class ProductLineMetrics(BaseModel):
    """Standardised customer-service metrics for one product line and quarter."""
    product_line: str
    product_name: str = ""
    orders: int = Field(description="Orders delivered in the period")
    tickets: int = Field(description="Support tickets raised in the period")
    contact_rate: float | None = Field(default=None, description="Tickets per 100 orders")
    csat: float | None = Field(default=None, description="Average CSAT, 1-5 scale")
    csat_change_qoq: float | None = Field(default=None, description="CSAT change vs previous quarter, points")
    first_contact_resolution: float | None = Field(default=None, description="% of tickets solved at first contact")
    repeat_contact_rate: float | None = Field(default=None, description="% of tickets needing more than one contact")
    avg_resolution_hours: float | None = Field(default=None, description="Average hours to resolve")
    return_rate: float | None = Field(default=None, description="% of orders returned")
    period: str | None = Field(default=None, description="Reported quarter; never inferred")


# ---------------------------------------------------------------------------
# Validation models
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Result of validating research findings."""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Report model (finance edition: ResearchReport / investment memo)
# ---------------------------------------------------------------------------

class CXResearchReport(BaseModel):
    """The final structured customer-experience research report."""
    title: str
    executive_summary: str
    product_metrics: list[ProductLineMetrics] = Field(default_factory=list)
    root_cause_analysis: str
    risk_factors: str
    recommendations: str
    conclusion: str
    sources: list[str] = Field(default_factory=list)
    confidence_score: float = Field(
        default=0.7, ge=0, le=1,
        description="Overall confidence in the report (0-1)",
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def check_api_key(key_name: str, required: bool = True) -> bool:
    """Check if an API key is set and print its status."""
    key = os.getenv(key_name)
    status = "+" if key else ("X" if required else "~")
    note = "configured" if key else ("not configured" if required else "not set (optional)")
    print(f"  [{status}] {key_name}: {note}")
    return bool(key)


def check_all_keys() -> bool:
    """Check the required API keys. Only the LLM key is mandatory in this edition."""
    print("Checking API keys...")
    llm_ok = check_api_key("OPENAI_API_KEY")
    check_api_key("TAVILY_API_KEY", required=False)
    print()
    return llm_ok


def persist_output(filename: str, content: str) -> str:
    """Save output to a file in the output directory."""
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved to {filepath}")
    return filepath


KNOWN_PRODUCT_LINES = {
    "FL-340": ["frostline", "refrigerator", "fridge"],
    "ZX1": ["zenith", "smartphone", "phone"],
    "AQP-500": ["aquapure", "purifier"],
    "HC-750": ["homechef", "mixer"],
    "BMX-15": ["breezemax", "air conditioner", " ac "],
}


def extract_product_line(text: str) -> str | None:
    """Extract a product-line code from a task description.

    Looks for codes like 'FL-340' first, then for product names ('FrostLine').
    Finance edition: extract_ticker('NVIDIA (NVDA)').
    """
    upper = text.upper()
    for code in KNOWN_PRODUCT_LINES:
        if re.search(rf"\b{re.escape(code)}\b", upper):
            return code
    low = f" {text.lower()} "
    for code, names in KNOWN_PRODUCT_LINES.items():
        if any(n in low for n in names):
            return code
    return None
