"""
tools/support_analytics.py
==========================
Structured customer-service metrics per product line (finance edition:
tools/financial_data.py, which called the Financial Datasets API).

Returns normalised ProductLineMetrics with consistent units:
- contact_rate in tickets per 100 orders
- CSAT on a 1-5 scale, change in points vs the previous quarter
- first_contact_resolution, repeat_contact_rate and return_rate as percentages

The data is a synthetic extract of NovaMart's support warehouse for
Q3 2026. To use real data, replace _WAREHOUSE with a query against your
ticketing system / data warehouse; keep the function signature.
"""

import asyncio

from common import ProductLineMetrics

_WAREHOUSE = {
    "FL-340": {"product_name": "FrostLine Refrigerator 340L", "orders": 3150, "tickets": 1134,
               "csat": 3.4, "csat_prev": 4.1, "first_contact_resolution": 48.0,
               "repeat_contact_rate": 38.0, "avg_resolution_hours": 71.0, "return_rate": 4.2},
    "ZX1": {"product_name": "Zenith X1 Smartphone", "orders": 18400, "tickets": 2760,
            "csat": 4.2, "csat_prev": 4.3, "first_contact_resolution": 74.0,
            "repeat_contact_rate": 14.0, "avg_resolution_hours": 19.0, "return_rate": 3.1},
    "AQP-500": {"product_name": "AquaPure RO Purifier", "orders": 5200, "tickets": 832,
                "csat": 4.4, "csat_prev": 4.3, "first_contact_resolution": 81.0,
                "repeat_contact_rate": 9.0, "avg_resolution_hours": 12.0, "return_rate": 1.8},
    "HC-750": {"product_name": "HomeChef Mixer Grinder", "orders": 9800, "tickets": 686,
               "csat": 4.3, "csat_prev": 4.2, "first_contact_resolution": 83.0,
               "repeat_contact_rate": 8.0, "avg_resolution_hours": 10.0, "return_rate": 2.4},
    "BMX-15": {"product_name": "BreezeMax Split AC 1.5T", "orders": 6100, "tickets": 1281,
               "csat": 3.9, "csat_prev": 4.0, "first_contact_resolution": 66.0,
               "repeat_contact_rate": 21.0, "avg_resolution_hours": 30.0, "return_rate": 1.2},
}
PERIOD = "Q3 2026"


async def get_product_line_metrics(product_line: str) -> ProductLineMetrics:
    """Fetch the quarter's service metrics for a product line.

    Args:
        product_line: Product-line code, e.g. 'FL-340', 'ZX1', 'AQP-500'.

    Returns:
        ProductLineMetrics with normalised data.

    Raises:
        ValueError: if the product line is unknown.
    """
    await asyncio.sleep(0.05)  # stands in for warehouse query latency
    code = product_line.strip().upper()
    row = _WAREHOUSE.get(code)
    if row is None:
        raise ValueError(f"Unknown product line: {product_line}")
    return ProductLineMetrics(
        product_line=code,
        product_name=row["product_name"],
        orders=row["orders"],
        tickets=row["tickets"],
        contact_rate=round(row["tickets"] / row["orders"] * 100, 1),
        csat=row["csat"],
        csat_change_qoq=round(row["csat"] - row["csat_prev"], 1),
        first_contact_resolution=row["first_contact_resolution"],
        repeat_contact_rate=row["repeat_contact_rate"],
        avg_resolution_hours=row["avg_resolution_hours"],
        return_rate=row["return_rate"],
        period=PERIOD,
    )
