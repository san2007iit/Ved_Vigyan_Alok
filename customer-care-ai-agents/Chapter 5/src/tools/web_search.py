"""
tools/web_search.py
===================
External voice of the customer: social media, review sites, consumer
forums and competitor moves (finance edition: financial news via Tavily).

If TAVILY_API_KEY is set, the tool runs a live Tavily web search. Without
a key it returns canned, synthetic results so the lab still runs with only
an OpenAI key. The canned results are clearly labelled as synthetic.
"""

import asyncio

from common import TAVILY_API_KEY

_CANNED = [
    (["frostline", "fl-340", "refrigerator", "fridge"], [
        "[Twitter, 2026-10-25] '@NovaMartCare 3 weeks without a working fridge after paying Rs 63k. "
        "Two failed repairs. #NovaMartFail' (1.2k reposts)",
        "[Consumer forum, 2026-10-18] Thread 'FrostLine 340L stops cooling after a month' has 46 replies; "
        "several users mention the technician 'did not have the part'.",
        "[Competitor news, 2026-10-10] Rival retailer CoolHome launched a '7-day no-questions replacement' "
        "guarantee for large appliances.",
    ]),
    (["zenith", "zx1", "smartphone", "phone", "delivery"], [
        "[Twitter, 2026-10-22] Several users report SwiftShip Delhi delays of 5-8 days during the festive season.",
        "[Review site, 2026-10] Zenith X1 rated 4.3/5; complaints focus on delivery time, not the product.",
    ]),
    (["aquapure", "aqp", "purifier"], [
        "[Review site, 2026-10] AquaPure RO rated 4.6/5; reviewers praise the filter-change reminder and video.",
    ]),
]


async def search_cx_web(query: str, max_results: int = 5) -> str:
    """Search the web and social media for customer sentiment and market context.

    Args:
        query: Search query, e.g. 'FrostLine refrigerator complaints'.
        max_results: Maximum results to return.

    Returns:
        Formatted text of results with sources.
    """
    if TAVILY_API_KEY:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
        resp = await client.search(query=query, max_results=max_results)
        rows = [f"[{r.get('url', '')}] {r.get('title', '')}: {r.get('content', '')[:300]}"
                for r in resp.get("results", [])]
        return "Web results (live):\n" + "\n".join(f"- {r}" for r in rows) if rows else "No web results."
    await asyncio.sleep(0.05)
    low = query.lower()
    hits = [r for keys, results in _CANNED if any(k in low for k in keys) for r in results]
    if not hits:
        return "No web results (synthetic mode: no canned results match this query)."
    return "Web results (SYNTHETIC - set TAVILY_API_KEY for live search):\n" + \
        "\n".join(f"- {r}" for r in hits[:max_results])
