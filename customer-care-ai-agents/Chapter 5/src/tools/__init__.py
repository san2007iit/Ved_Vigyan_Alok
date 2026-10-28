"""
tools/
======
Customer-experience data tools for the Deep Search Agent.

Each module wraps a specific data source and returns clean,
normalised data that the agent can reason about directly.
"""

from tools.cx_documents import get_cx_documents
from tools.support_analytics import get_product_line_metrics
from tools.web_search import search_cx_web

__all__ = [
    "get_product_line_metrics",
    "get_cx_documents",
    "search_cx_web",
]
