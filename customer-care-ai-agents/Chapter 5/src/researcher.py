"""
researcher.py
=============
Plan execution component for the Deep Search Agent.

Executes a ResearchPlan by running sub-tasks according to their dependency
graph. Independent tasks execute in parallel with asyncio.gather();
dependent tasks wait for their prerequisites.
"""

import asyncio

from pydantic_ai import Agent

from common import SYNTHESIS_MODEL, ResearchPlan, SubTask, TaskStatus, extract_product_line
from tools.cx_documents import get_cx_documents
from tools.support_analytics import get_product_line_metrics
from tools.web_search import search_cx_web

# PydanticAI agent for analysis tasks
analyst_agent = Agent(
    model=SYNTHESIS_MODEL,
    system_prompt=(
        "You are a senior customer-experience analyst. Analyse the provided data and produce "
        "a clear, factual response. Use specific numbers from the data. Do not fabricate data: "
        "if information is missing, state that explicitly. Separate symptoms (e.g. low CSAT) "
        "from root causes (e.g. a missing spare part)."
    ),
    retries=1,
)


async def execute_plan(plan: ResearchPlan) -> dict[int, str]:
    """Execute a research plan, respecting task dependencies.

    Tasks with no unmet dependencies execute in parallel. Each wave of
    parallel tasks completes before the next wave of now-unblocked tasks begins.
    """
    results: dict[int, str] = {}
    tasks_by_id = {t.id: t for t in plan.sub_tasks}

    while any(t.status == TaskStatus.PENDING for t in plan.sub_tasks):
        ready = [
            t for t in plan.sub_tasks
            if t.status == TaskStatus.PENDING
            and all(tasks_by_id[dep].status == TaskStatus.COMPLETED for dep in t.dependencies)
        ]
        if not ready:
            for t in plan.sub_tasks:
                if t.status == TaskStatus.PENDING:
                    t.status = TaskStatus.FAILED
                    t.result = "Could not execute: unresolved dependencies"
                    results[t.id] = t.result
            break

        async def _execute_one(task: SubTask) -> tuple[int, str]:
            task.status = TaskStatus.IN_PROGRESS
            try:
                result = await _route_and_execute(task, results)
                if not isinstance(result, str) or not result.strip():
                    raise ValueError("Task returned no evidence")
                task.status = TaskStatus.COMPLETED
                task.result = result
                return task.id, result
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.result = f"[FAILED] Task {task.id}: {e}"
                return task.id, task.result

        completed = await asyncio.gather(*[_execute_one(t) for t in ready])
        for task_id, result in completed:
            results[task_id] = result
            print(f"    [{tasks_by_id[task_id].status.value}] Task {task_id}: "
                  f"{tasks_by_id[task_id].description[:60]}")
    return results


async def _route_and_execute(task: SubTask, prior_results: dict[int, str]) -> str:
    """Route a sub-task to the appropriate tool(s) and execute it."""
    context = ""
    if task.dependencies:
        context = "\n\n---\n\n".join(prior_results[d] for d in task.dependencies if d in prior_results)

    sources = task.data_sources

    # Support analytics: structured metrics
    if "support_analytics" in sources and not task.dependencies:
        code = extract_product_line(task.description)
        if code:
            metrics = await get_product_line_metrics(code)
            return metrics.model_dump_json(indent=2)

    # Internal documents: qualitative root-cause evidence
    if "cx_documents" in sources and "support_analytics" not in sources:
        code = extract_product_line(task.description)
        if code:
            return await get_cx_documents(code)

    # Web / social search
    if "web_search" in sources and len(sources) == 1:
        return await search_cx_web(task.description)

    # Mixed sources or analysis tasks: reason over prior findings with the LLM
    return await _llm_analyze(task.description, context)


async def _llm_analyze(description: str, context: str) -> str:
    """Use the analyst agent for tasks that need reasoning rather than a tool call."""
    prompt = f"Task: {description}\n\nAvailable data:\n{context if context else 'No prior data.'}"
    result = await analyst_agent.run(prompt)
    return result.output
