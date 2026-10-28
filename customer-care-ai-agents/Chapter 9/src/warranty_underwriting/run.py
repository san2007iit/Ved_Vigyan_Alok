"""
run.py
======
Pricing a NovaMart Care+ extended warranty with parallel data enrichment.

Usage (from the Chapter 9 folder):
    python -m src.warranty_underwriting.run

A sequential workflow whose enrichment stage applies the
parallelization style (Ch 7, Section 2.5).
"""

import asyncio
import time


async def check_product_reliability(sku: str) -> dict:
    """Simulate the service-analytics lookup (parallel enrichment agent 1)."""
    await asyncio.sleep(1.0)  # Simulate API latency
    return {"sku": sku, "failure_rate_12m": 0.048,
            "avg_repair_cost_inr": 4200, "reliability_band": "good"}


async def check_claims_history(customer_id: str) -> dict:
    """Simulate the CRM warranty-claims lookup (parallel enrichment agent 2)."""
    await asyncio.sleep(1.2)
    return {"customer_id": customer_id, "warranty_claims_3yr": 1,
            "return_rate_12m": 0.0, "abuse_flag": False}


async def check_service_area(pincode: str) -> dict:
    """Simulate the service-area risk lookup (parallel enrichment agent 3)."""
    await asyncio.sleep(0.8)
    return {"pincode": pincode, "power_fluctuation": "high",
            "technician_coverage": "full", "climate": "extreme heat"}


async def enrich_application(application: dict) -> dict:
    """Run three enrichment agents concurrently."""
    print("Running parallel data enrichment...")
    reliability, history, area = await asyncio.gather(
        check_product_reliability(application["sku"]),
        check_claims_history(application["customer_id"]),
        check_service_area(application["pincode"]),
    )
    print(f"  Product failure rate (12m): {reliability['failure_rate_12m']:.1%}")
    print(f"  Customer warranty claims (3y): {history['warranty_claims_3yr']}")
    print(f"  Service area: power fluctuation {area['power_fluctuation']}, "
          f"technicians {area['technician_coverage']}")
    return {"reliability": reliability, "history": history, "area": area}


def assess_risk(enrichment: dict, device_value: float) -> dict:
    """Synthesize enrichment data into a risk score, annual premium and decision."""
    failure_rate = enrichment["reliability"]["failure_rate_12m"]
    claims = enrichment["history"]["warranty_claims_3yr"]
    power = enrichment["area"]["power_fluctuation"]

    # Simplified risk scoring
    base_score = 100
    if failure_rate <= 0.05:
        base_score -= 10
    elif failure_rate > 0.08:
        base_score += 20
    base_score += claims * 15
    if power == "high":
        base_score += 10           # voltage swings damage compressors and boards
    if enrichment["history"]["abuse_flag"]:
        base_score += 40

    # Premium calculation (simplified)
    base_premium = device_value * 0.04  # 4% of the device value per year
    risk_multiplier = 1.0 + (base_score - 80) / 100

    return {
        "risk_score": base_score,
        "risk_tier": ("preferred" if base_score < 70
                      else "standard" if base_score < 100
                      else "high_risk"),
        "annual_premium_inr": round(base_premium * risk_multiplier, 2),
        "decision": ("accept" if base_score < 120
                     else "refer_to_senior"),
    }


async def run_underwriting():
    """Run the complete Care+ pricing pipeline."""
    application = {
        "application_id": "CARE-APP-2026-014",
        "customer_id": "C1002",
        "name": "Rahul Verma",
        "product": "BreezeMax 1.5 ton inverter AC",
        "sku": "BMX-15",
        "device_value_inr": 46000,
        "pincode": "110017",
        "plan_requested": "Care+ 3-year extended warranty",
    }
    print(f"Pricing Care+ for: {application['name']} - {application['product']}\n")

    # Step 1: Parallel enrichment
    start = time.perf_counter()
    enrichment = await enrich_application(application)
    print(f"  (enrichment took {time.perf_counter() - start:.1f}s, not 3.0s)")

    # Step 2: Risk assessment
    print("\nAssessing risk...")
    risk = assess_risk(enrichment, application["device_value_inr"])

    print(f"\n{'=' * 50}\n  CARE+ PRICING DECISION\n{'=' * 50}")
    print(f"  Risk score: {risk['risk_score']}")
    print(f"  Risk tier: {risk['risk_tier']}")
    print(f"  Annual premium: Rs {risk['annual_premium_inr']:,.2f}")
    print(f"  Decision: {risk['decision']}")
    return risk


if __name__ == "__main__":
    asyncio.run(run_underwriting())
