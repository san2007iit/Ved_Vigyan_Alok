"""
planner.py
==========
Research planning component for the Deep Search Agent.

Takes a research question and produces a ResearchPlan: a structured set of
sub-tasks with explicit dependencies. Uses PydanticAI's structured output
to guarantee a valid plan schema.
"""

from pydantic_ai import Agent

from common import PLANNING_MODEL, ResearchPlan

PLANNER_SYSTEM_PROMPT = """\
You are a customer-experience research planner at a retailer. Given a research
question, decompose it into concrete sub-tasks that can be executed by tools.

Available data sources for each sub-task:
- support_analytics: Structured service metrics per product line (orders, tickets,
  contact rate, CSAT and its change, first-contact resolution, repeat-contact rate,
  resolution time, return rate). Best for quantitative data.
- cx_documents: Internal documents per product line (engineering known-issue bulletins,
  field-service reports, QA call audits). Best for root causes.
- web_search: Social media, review sites, consumer forums, competitor moves.
  Best for external sentiment and market context.

Rules for creating the plan:
1. Each sub-task must have a clear, measurable outcome (e.g., "Fetch FL-340 CSAT,
   contact rate and FCR for Q3" not "Understand the fridge problem").
2. Name the product-line code (e.g. FL-340, ZX1, AQP-500) in every product-specific task.
3. Specify which data_sources each sub-task should use (can be multiple).
4. Mark dependencies: which task IDs must complete before this task can start.
5. Tasks with no dependencies can execute in parallel: maximise parallelism.
6. For multi-product comparisons, create parallel tasks for each product line.
7. Include a comparative analysis task that depends on all product data tasks.
8. The final task should synthesise all findings: it depends on everything.
9. Use sequential IDs starting from 1.
10. Aim for 5-10 sub-tasks. Too few = insufficient depth. Too many = overhead.
11. If the question is about a single product line, still gather a comparison line.\
"""

# PydanticAI agent with structured output
planner_agent = Agent(
    model=PLANNING_MODEL,
    system_prompt=PLANNER_SYSTEM_PROMPT,
    output_type=ResearchPlan,
    retries=2,
)


async def create_research_plan(question: str, additional_context: str = "") -> ResearchPlan:
    """Generate a structured research plan from a question.

    Args:
        question: The research question to plan for.
        additional_context: Optional context from previous attempts
            (e.g., "Previous attempt failed to get ZX1 data").

    Returns:
        A ResearchPlan with sub-tasks and dependency information.
    """
    user_message = question
    if additional_context:
        user_message += f"\n\nAdditional context:\n{additional_context}"

    result = await planner_agent.run(user_message)
    plan = result.output
    _validate_plan(plan)
    return plan


def _validate_plan(plan: ResearchPlan) -> None:
    """Check that the plan is semantically sound (PydanticAI already checked its shape).

    - At least 2 sub-tasks
    - All dependency IDs reference existing tasks
    - No task depends on itself
    - At least one task has no dependencies (something can start)
    """
    task_ids = {t.id for t in plan.sub_tasks}
    if len(plan.sub_tasks) < 2:
        raise ValueError(f"Plan has only {len(plan.sub_tasks)} sub-tasks. A research plan needs at least 2.")
    for task in plan.sub_tasks:
        for dep in task.dependencies:
            if dep not in task_ids:
                raise ValueError(f"Task {task.id} depends on non-existent task {dep}")
            if dep == task.id:
                raise ValueError(f"Task {task.id} depends on itself (circular dependency)")
    if not [t for t in plan.sub_tasks if not t.dependencies]:
        raise ValueError("No tasks can start - all have dependencies (deadlock)")


def format_plan(plan: ResearchPlan) -> str:
    """Format a research plan for display."""
    lines = [f"Research Plan: {plan.question}", f"Sub-tasks: {len(plan.sub_tasks)}", "-" * 50]
    for task in plan.sub_tasks:
        deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
        lines.append(f"  [{task.id}] {task.description}\n       Sources: {', '.join(task.data_sources)}{deps}")
    return "\n".join(lines)
