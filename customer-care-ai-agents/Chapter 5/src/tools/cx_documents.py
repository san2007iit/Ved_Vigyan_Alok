"""
tools/cx_documents.py
=====================
Qualitative internal intelligence per product line (finance edition:
tools/sec_filings.py, which read risk factors from SEC EDGAR).

Three document types matter for root-cause analysis in customer care:
- engineering known-issue bulletins (KIB),
- field-service reports from technicians,
- QA call-audit findings from the contact centre.

These synthetic excerpts stand in for a document store (SharePoint,
Confluence, the QA platform). Replace _DOCS with a search over your own
repository; keep the function signature.
"""

import asyncio

_DOCS = {
    "FL-340": [
        "[KIB-2026-014, Engineering, 2026-09-02] Compressor relay part R-22 shows a 3.1% early-failure "
        "rate in FL-340 units built July-August 2026 (supplier batch B7). Symptom: unit stops cooling "
        "1-6 weeks after installation. Spare R-22 is back-ordered until 15 November 2026.",
        "[Field-service report, Q3 2026] First-visit fix rate for FL-340 cooling complaints is 52% "
        "(norm: 85%). In 41% of failed first visits the technician lacked part R-22. Average 2.3 visits "
        "per cooling case.",
        "[QA call audit, Q3 2026, n=120 FL-340 calls] 40% of escalated calls include the phrase "
        "'explain again'; agents have no script for repeat failures and rarely offer the replacement "
        "allowed by the repeat-failure rule (policy KB-102) until the third contact.",
    ],
    "ZX1": [
        "[Logistics report, Q3 2026] 61% of ZX1 tickets are 'where is my order' contacts driven by the "
        "SwiftShip Delhi (Okhla) hub backlog during the festive surge; product-defect tickets are flat.",
        "[KIB-2026-011, Engineering] Software update 2.1 caused minor battery-drain complaints "
        "(0.4% of units); fixed by update 2.1.1 on 2026-09-20.",
    ],
    "AQP-500": [
        "[CX initiative, August 2026] A filter-change how-to video and WhatsApp reminder launched "
        "in August cut 'how-to' tickets by 35% and raised first-contact resolution from 74% to 81%.",
    ],
    "HC-750": ["[QA call audit, Q3 2026] No systemic issues; most tickets are warranty-registration questions."],
    "BMX-15": [
        "[Field-service report, Q3 2026] Installation-slot availability fell in Chennai due to rain; "
        "two smoke/burning-smell incidents were escalated to Product Safety and are under investigation.",
    ],
}


async def get_cx_documents(product_line: str) -> str:
    """Return internal known-issue, field-service and QA findings for a product line.

    Args:
        product_line: Product-line code, e.g. 'FL-340'.

    Returns:
        A text block of document excerpts with their identifiers.
    """
    await asyncio.sleep(0.05)
    code = product_line.strip().upper()
    docs = _DOCS.get(code)
    if not docs:
        raise ValueError(f"No internal documents for product line {product_line}")
    return f"Internal CX documents for {code}:\n" + "\n".join(f"- {d}" for d in docs)
