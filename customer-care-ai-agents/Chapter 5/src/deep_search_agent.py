"""
deep_search_agent.py
====================
Main orchestrator for the Deep Search Agent - Chapter 5 Lab (customer-care edition).

Runs the full deep-research pipeline:

    Plan -> Execute -> Validate -> (Replan?) -> Synthesize

Usage:
    python deep_search_agent.py
    python deep_search_agent.py --question "Why are AC installation complaints rising?" --products BMX-15,AQP-500

Requires OPENAI_API_KEY (TAVILY_API_KEY optional for live web search).
"""

import argparse
import asyncio
import sys
from datetime import datetime

from common import MAX_REPLAN_ATTEMPTS, CXResearchReport, check_all_keys, persist_output
from planner import create_research_plan, format_plan
from researcher import execute_plan
from synthesizer import format_report_markdown, synthesize_report
from validator import format_validation, validate_research

DEFAULT_QUESTION = (
    "Why did customer satisfaction for the FrostLine refrigerator line (FL-340) fall in Q3 2026? "
    "Compare it with the Zenith X1 smartphone (ZX1) and the AquaPure purifier (AQP-500) on CSAT, "
    "contact rate, first-contact resolution and repeat-contact rate, identify root causes and "
    "risks, and produce a customer-experience research memo with recommendations."
)
DEFAULT_PRODUCT_LINES = ["FL-340", "ZX1", "AQP-500"]


async def deep_search(question: str, product_lines: list[str], verbose: bool = True) -> CXResearchReport:
    """Run Plan -> Execute -> Validate -> Synthesize with up to MAX_REPLAN_ATTEMPTS replans."""
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"\n{'=' * 60}\n  DEEP SEARCH AGENT (customer experience)\n{'=' * 60}")
    log(f"\n  Question:      {question}\n  Product lines: {', '.join(product_lines)}\n{'=' * 60}\n")

    additional_context = ""
    for attempt in range(MAX_REPLAN_ATTEMPTS + 1):
        # Phase 1: PLAN
        log(f"[Phase 1] Planning (attempt {attempt + 1}/{MAX_REPLAN_ATTEMPTS + 1})...")
        try:
            plan = await create_research_plan(question, additional_context)
        except Exception as e:
            log(f"  [ERROR] Planning failed: {e}")
            if attempt < MAX_REPLAN_ATTEMPTS:
                additional_context = f"Previous planning attempt failed: {e}"
                continue
            raise
        parallel = sum(1 for t in plan.sub_tasks if not t.dependencies)
        log(f"  Generated {len(plan.sub_tasks)} sub-tasks ({parallel} parallel, "
            f"{len(plan.sub_tasks) - parallel} sequential)\n")
        if verbose:
            print(format_plan(plan) + "\n")

        # Phase 2: EXECUTE
        log("[Phase 2] Executing research plan...\n")
        results = await execute_plan(plan)
        log(f"\n  Completed {len(results)} tasks\n")

        # Phase 3: VALIDATE
        log("[Phase 3] Validating findings...\n")
        validation = validate_research(plan, results, product_lines)
        if verbose:
            print(format_validation(validation) + "\n")
        if validation.is_valid:
            log("  Validation passed - proceeding to synthesis\n")
            break
        if attempt < MAX_REPLAN_ATTEMPTS:
            issues = validation.gaps + validation.errors
            additional_context = ("Previous research attempt had the following issues that need to be "
                                  "addressed:\n" + "\n".join(f"- {g}" for g in issues))
            log(f"  Replanning to address {len(issues)} issue(s)...\n")
        else:
            raise RuntimeError("Research validation failed after all replans; synthesis blocked: "
                               + "; ".join(validation.errors + validation.gaps))

    # Phase 4: SYNTHESIZE
    log("[Phase 4] Synthesizing research report...\n")
    report = await synthesize_report(question, results, validation)
    log(f"{'=' * 60}\n  {report.title}\n{'=' * 60}\n")
    log(report.executive_summary)
    log(f"\n  Confidence: {report.confidence_score:.0%}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Search Agent - customer-experience research")
    parser.add_argument("--question", "-q", type=str, default=DEFAULT_QUESTION, help="Research question")
    parser.add_argument("--products", "-p", type=str, default=",".join(DEFAULT_PRODUCT_LINES),
                        help="Comma-separated product-line codes that must be covered (e.g. FL-340,ZX1)")
    parser.add_argument("--output-format", choices=["markdown", "json", "both"], default="both")
    parser.add_argument("--quiet", "-Q", action="store_true", help="Only print the final report")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if not check_all_keys():
        print("\nMissing OPENAI_API_KEY. Put it in a .env file (TAVILY_API_KEY is optional).\n")
        sys.exit(1)
    products = [p.strip().upper() for p in args.products.split(",")]
    report = await deep_search(args.question, products, verbose=not args.quiet)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(products)
    if args.output_format in ("markdown", "both"):
        print(f"\n  Markdown report: {persist_output(f'report_{tag}_{stamp}.md', format_report_markdown(report))}")
    if args.output_format in ("json", "both"):
        print(f"  JSON report:     {persist_output(f'report_{tag}_{stamp}.json', report.model_dump_json(indent=2))}")
    print("\n  Deep Search Agent completed successfully.\n")


if __name__ == "__main__":
    asyncio.run(main())
