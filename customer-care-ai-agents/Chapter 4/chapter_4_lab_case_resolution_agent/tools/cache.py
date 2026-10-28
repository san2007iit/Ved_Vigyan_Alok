"""
cache.py
--------
Simple in-memory cache for tool results within a single pipeline run.

Each entry is keyed by a tuple (tool_name, *args) and stores the JSON
string that the tool would otherwise rebuild from the CRM. The cache is
populated on the first call and reused on all subsequent calls within the
same run, eliminating redundant CRM requests when several agents call the
same tool with the same inputs (for example, both the Customer Profile
Agent and the Churn Risk Agent read the ticket history).

Call clear() at the start of each analyze_case() call so every new case
starts from fresh data.
"""

_cache: dict[tuple, str] = {}


def get(key: tuple) -> str | None:
    return _cache.get(key)


def store(key: tuple, value: str) -> None:
    _cache[key] = value


def clear() -> None:
    _cache.clear()
