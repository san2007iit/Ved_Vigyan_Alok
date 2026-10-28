import copy
import json
import os
import re

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

MODEL = os.getenv("CC_ANTHROPIC_MODEL", "claude-opus-5-5")

# ---------- Pure environment ----------
## A mock contact-centre simulation. In practice you would plug in your workforce-management
## (WFM) system's queue model (Erlang C) and your real staffing costs.
QUEUES = ["billing", "returns", "technical", "delivery", "installations"]

INITIAL_QUEUE_STATE = {
    "avg_wait_min": 40.0,        # current average customer wait across queues
    "target_min": 30.0,          # service-level target
    "extra_cost_k_inr": [],      # extra daily staffing cost of each action, in Rs thousand
}


def apply_action(state, queues, pct):
    """Apply a staffing action to a queue state and return the new snapshot.

    Args:
        state: dict with `avg_wait_min`, `target_min` and `extra_cost_k_inr`
            (list of floats). Not mutated: a deep copy is returned.
        queues: list of queue names receiving extra capacity.
        pct: percentage of extra capacity added to each named queue
            (overtime, cross-skilled agents, or callback deflection).

    Returns:
        (new_state, observation_text)

    Raises:
        ValueError: for unknown queues or an implausible percentage. The environment,
        not the LLM, enforces what is physically possible.
    """
    unknown = [q for q in queues if q not in QUEUES]
    if unknown:
        raise ValueError(f"unknown queue(s) {unknown}; valid: {QUEUES}")
    if not 0 < pct <= 25:
        raise ValueError(f"percent_each must be in (0, 25], got {pct}")
    new = copy.deepcopy(state)
    cost = 5 + len(queues) * 2 + pct * 1.5
    new["extra_cost_k_inr"].append(cost)
    new["avg_wait_min"] -= pct * len(queues) * 0.4
    obs = (
        f"Added {pct}% capacity to {queues}. Extra cost Rs {cost:.1f}k per day. "
        f"New average wait: {new['avg_wait_min']:.1f} min."
    )
    return new, obs


# ---------- Claude Agent SDK querying ----------
async def _run_query(prompt, *, system_prompt, allowed_tools, max_turns, use_skills=False):
    options_kwargs = dict(
        model=MODEL,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
    )
    if use_skills:
        options_kwargs["setting_sources"] = ["project"]   # loads .claude/skills/* from this folder
    options = ClaudeAgentOptions(**options_kwargs)

    chunks = []
    async for msg in query(prompt=prompt, options=options):
        if not isinstance(msg, AssistantMessage):
            continue
        if msg.parent_tool_use_id is not None:
            continue
        for block in msg.content:
            if isinstance(block, TextBlock) and block.text:
                chunks.append(block.text)
    return "\n".join(chunks)


def _trajectory_block(parent_state, proposal, child_state, observation):
    return (
        f"Trajectory:\n"
        f"- Starting average wait: {parent_state['avg_wait_min']:.1f} min\n"
        f"- Action: add {proposal['percent_each']}% capacity to {proposal['queues']}\n"
        f"- Rationale: {proposal.get('rationale', '')}\n"
        f"- Observation: {observation}\n"
        f"- Resulting average wait: {child_state['avg_wait_min']:.1f} min\n"
        f"- Target: {child_state['target_min']} min"
    )


# ---------- LATS Step 2: one-shot k-expand ----------
PROPOSE_SYSTEM_PROMPT = f"""You are a contact-centre workforce planner reducing customer wait time.

You will be asked to propose K distinct staffing actions from a given queue state.
Valid queues: {QUEUES}.
The K candidates MUST differ materially from one another: different queue selection,
different capacity size, or different concentration. Mode collapse to identical or
near-identical actions is a failure.

Output exactly one fenced JSON code block and no other text. The block must contain a
JSON array of K objects, each with these fields:
- queues: list of queue names (strings, non-empty, from the valid list)
- percent_each: number, the % extra capacity added to each named queue (0-25)
- rationale: short string explaining the candidate's distinct angle

Example:
```json
[
  {{"queues": ["returns", "technical"], "percent_each": 10.0, "rationale": "focus on the two longest queues"}},
  {{"queues": ["billing", "returns", "technical", "delivery"], "percent_each": 5.0, "rationale": "spread small overtime"}}
]
```
"""


async def propose_k_actions(state, k_expand, lessons):
    """One LM invocation that returns k distinct staffing proposals.

    Because the model writes all k proposals in one response, candidate i is generated with
    candidates 1..i-1 already in its context: the paper's mechanism for diversity.
    Returns a list of dicts with keys `queues`, `percent_each`, `rationale` ([] if unparseable).
    """
    lessons_block = ""
    if lessons:
        lessons_block = "\n\nLessons from prior attempts:\n- " + "\n- ".join(lessons)
    prompt = (
        f"Current average wait: {state['avg_wait_min']:.1f} min. "
        f"Target: {state['target_min']} min. "
        f"K = {k_expand}. Propose {k_expand} materially distinct staffing actions."
        f"{lessons_block}"
    )
    text = await _run_query(prompt, system_prompt=PROPOSE_SYSTEM_PROMPT, allowed_tools=[], max_turns=2)
    return _parse_proposals(text)


def _parse_proposals(text):
    """Pull a JSON array of proposals from the model's output. [] on failure."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    if not candidate.startswith("["):
        a, b = candidate.find("["), candidate.rfind("]")
        if a == -1 or b <= a:
            return []
        candidate = candidate[a:b + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [
        d for d in data
        if isinstance(d, dict)
        and isinstance(d.get("queues"), list) and d["queues"]
        and all(isinstance(q, str) for q in d["queues"])
        and isinstance(d.get("percent_each"), (int, float))
    ]


# ---------- LATS Step 3: per-child evaluation ----------
EVAL_SYSTEM_PROMPT = """You are evaluating a single staffing trajectory for a contact centre.

Use the Skill tool to invoke `staffing-evaluator` on the trajectory described.
End your response with a line of the exact form: SCORE: <integer 1-10>
"""


async def evaluate_proposal(parent_state, proposal, child_state, observation):
    """Score a trajectory with the staffing-evaluator skill. Returns a score in [0, 1] or None."""
    text = await _run_query(
        _trajectory_block(parent_state, proposal, child_state, observation) + "\n\nEvaluate this trajectory.",
        system_prompt=EVAL_SYSTEM_PROMPT,
        allowed_tools=["Skill"],
        max_turns=4,
        use_skills=True,
    )
    return _parse_score(text)


def _parse_score(text):
    """Integer after 'SCORE:' normalised to [0, 1]; the LAST valid line wins. None on failure."""
    last = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("SCORE:"):
            continue
        match = re.fullmatch(r"SCORE:\s*(10|[1-9])", stripped, re.IGNORECASE)
        if match:
            last = int(match.group(1)) / 10.0
    return last


# ---------- LATS Step 6: reflection on failure ----------
REFLECT_SYSTEM_PROMPT = """You are reflecting on a staffing trajectory that fell short of its target.

Use the Skill tool to invoke `staffing-reflector` on the trajectory described, then
end your response with a line of the exact form:
LESSON: When <situation>, do <action>.
"""


async def reflect_on_failure(parent_state, proposal, child_state, observation):
    """Generate a `When ..., do ...` lesson with the staffing-reflector skill (None if unusable)."""
    text = await _run_query(
        _trajectory_block(parent_state, proposal, child_state, observation)
        + "\n\nThe target was not reached. Reflect on this trajectory.",
        system_prompt=REFLECT_SYSTEM_PROMPT,
        allowed_tools=["Skill"],
        max_turns=4,
        use_skills=True,
    )
    return _parse_lesson(text)


def _parse_lesson(text):
    """Pull the 'LESSON: When ..., do ...' line; skip template placeholders; LAST valid wins."""
    last = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("LESSON:"):
            continue
        body = stripped.split(":", 1)[1].strip()
        if not body.lower().startswith("when "):
            continue
        if "<situation>" in body.lower() or "<action>" in body.lower():
            continue
        last = body
    return last


# ---------- Output utilities ----------
def print_tree(node, indent=0):
    pad = "  " * indent
    flag = " [TERMINAL]" if node.is_terminal else ""
    print(f"{pad}{node.action_hint}: N={node.visits} V={node.value:.2f} "
          f"wait={node.state['avg_wait_min']:.1f}min{flag}")
    for c in node.children:
        print_tree(c, indent + 1)


def best_leaf(root):
    """DFS: return the leaf node with the highest value."""
    best = root
    stack = [root]
    while stack:
        n = stack.pop()
        if not n.children and n.value > best.value:
            best = n
        stack.extend(n.children)
    return best
