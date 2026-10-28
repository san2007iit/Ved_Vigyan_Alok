"""lab2_guardrails.py : A guardrail pipeline around the case-resolution workflow.

Six steps assemble the input, tool-output, and output guardrails from guardrails.py on the
OpenAI Agents SDK's native guardrail hooks, then red-team the result. The chapter's four actions
map onto the SDK like this: 
1- BLOCK = an input/output guardrail tripwire; 
2- ESCALATE = a tool-output guardrail that raises on retrieved content; 
3- REDACT = a deterministic transform (the SDK's guardrails are tripwires and cannot rewrite content); 
4- ALLOW = the guardrail passes.

Run the whole lab:  python lab2_guardrails.py
Run one step:       python -c "from lab2_guardrails import step_6_red_team; step_6_red_team()"
"""

from __future__ import annotations

import asyncio
import io
import os
import sys

from agents import Runner
from agents.exceptions import (InputGuardrailTripwireTriggered,
                               OutputGuardrailTripwireTriggered,
                               ToolOutputGuardrailTripwireTriggered)

import care_workflow as cw
import guardrails as gr
import redteam_cases as rt

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":  # idempotent: wrap once
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "results", "lab_runs")

# -- Guarded Orchestrator : Orchestrator agent based on care_workflow.py + guardrails.py -----
_GUARDED = gr.build_guarded_orchestrator()

# -- Saving all results in a local folder --------------------------------------------------
def _report(name: str, body: str) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{name}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"  -> wrote {os.path.relpath(path, ROOT)}")


async def _run_guarded(customer_id: str, note: str, poisoned_messages: str | None):
    """Send one request through the full guarded workflow. PII (Personally Identifiable 
    Information) redaction is a transform applied before the run; the input, 
    tool-output, and output guardrails are the SDK tripwires. 
    Returns which stage fired (if any) and the served answer."""
    redacted_note, was_redacted = gr.redact_pii(note or "")
    fired = {"pii": "REDACT"} if was_redacted else {}

    saved = None
    if poisoned_messages is not None:
        saved, cw._fetch_messages = cw._fetch_messages, lambda c, n: poisoned_messages
    prompt = (f"{redacted_note}\nResolve the open case for customer {customer_id}. "
              f"Gather the order status and the customer's messages, then recommend a resolution.")
    try:
        result = await Runner.run(_GUARDED, prompt, max_turns=cw.MAX_TURNS)
        answer, appended = gr.append_disclaimer(result.final_output)
        if appended:
            fired["output_policy"] = "REDACT"
        return {"served": True, "blocked_at": None, "fired": fired, "answer": answer}
    except InputGuardrailTripwireTriggered:
        fired["input_validation"] = "BLOCK"
        return {"served": False, "blocked_at": "input_validation", "fired": fired, "answer": None}
    except ToolOutputGuardrailTripwireTriggered:
        fired["injection"] = "ESCALATE"
        return {"served": False, "blocked_at": "injection", "fired": fired, "answer": None}
    except OutputGuardrailTripwireTriggered:
        fired["output_policy"] = "BLOCK"
        return {"served": False, "blocked_at": "output_policy", "fired": fired, "answer": None}
    finally:
        if saved is not None:
            cw._fetch_messages = saved


# -- Step 1: The pipeline skeleton - End-to-end normal run--------------------------------
def step_1_pipeline_skeleton() -> dict:
    """Run one legitimate request through the guarded workflow and show it is served with the
    disclaimer appended : The ALLOW/REDACT path end to end."""
    out = asyncio.run(_run_guarded("C1001", "Please review this case.", None))
    lines = [f"served: {out['served']}  blocked_at: {out['blocked_at']}",
             f"fired:  {out['fired']}"]
    text = "\n".join(lines)
    print(text)
    _report("oai_step_1_guard_pipeline_skeleton", f"# Step 1 -- pipeline skeleton\n\n```\n{text}\n```\n")
    return out


# -- Step 2: Input validation (BLOCK via input guardrail) ---------------------------------
def step_2_input_validation() -> list:
    """A customer ID outside the desk's allow-list trips the input guardrail and blocks the run before any model
    or tool call. A listed customer passes."""
    rows = []
    for customer_id in ("C1001", "C9999"):
        out = asyncio.run(_run_guarded(customer_id, "Please review.", None))
        rows.append((customer_id, out["blocked_at"] or "served"))
    text = "\n".join(f"{t:<7}{status}" for t, status in rows)
    print(text)
    _report("oai_step_2_guard_input_validation", f"# Step 2 -- input validation\n\n```\n{text}\n```\n")
    return rows


# -- Step 3: PII redaction (REDACT transform) ---------------------------------------------
def step_3_pii_redaction() -> None:
    """Redact Aadhaar, phone, account numbers and emails from an agent note before the workflow sees it."""
    note = "Aadhaar 2345 6789 0123, call +91 98765 43210, refund to a/c 123456789012, cc jane.doe@example.com."
    redacted, was = gr.redact_pii(note)
    text = f"in:  {note}\nout: {redacted}\nredacted: {was}"
    print(text)
    _report("oai_step_3_guard_pii_redaction", f"# Step 3 -- PII redaction\n\n```\n{text}\n```\n")


# -- Step 4: Injection defence (ESCALATE via tool-output guardrail) -----------------------
def step_4_injection_defence() -> dict:
    """Screen the customer's messages for indirect injection. Show the two-layer check flagging a
    poisoned note and passing a clean one, then confirm the tool-output guardrail escalates
    a poisoned run end to end."""
    poisoned = gr.evaluate_injection(rt.POISONED_MESSAGES)
    clean = gr.evaluate_injection(rt.CLEAN_MESSAGES)
    end_to_end = asyncio.run(_run_guarded("C1002", "Please review.", rt.POISONED_MESSAGES))
    lines = [f"poisoned note -> injection detected: {poisoned}",
             f"clean note    -> injection detected: {clean}",
             f"end-to-end poisoned run blocked_at: {end_to_end['blocked_at']}"]
    text = "\n".join(lines)
    print(text)
    _report("oai_step_4_guard_injection_defence", f"# Step 4 -- injection defence\n\n```\n{text}\n```\n")
    return {"poisoned": poisoned, "clean": clean, "end_to_end": end_to_end["blocked_at"]}


# -- Step 5: Output policy (BLOCK via output guardrail) -----------------------------------
def step_5_output_policy() -> None:
    """Apply the output policy to three sample answers: clean, a forbidden phrase, and an
    off-vocabulary resolution."""
    samples = {
        "clean": "A replacement is warranted under KB-102.\nRESOLUTION: REPLACE\nCONFIDENCE: HIGH",
        "forbidden_phrase": "We guarantee a full refund.\nRESOLUTION: REFUND\nCONFIDENCE: HIGH",
        "off_vocab_resolution": "Give them everything.\nRESOLUTION: GIVE_EVERYTHING\nCONFIDENCE: HIGH",
    }
    lines = []
    for label, ans in samples.items():
        info = gr.evaluate_output_policy(ans)
        action = "BLOCK" if info["tripped"] else "ALLOW"
        reason = info["forbidden_phrase"] or (f"resolution {info['resolution']!r}" if info["off_vocab"] else "ok")
        lines.append(f"{label:<18}{action:<8}{reason}")
    text = "\n".join(lines)
    print(text)
    _report("oai_step_5_guard_output_policy", f"# Step 5 -- output policy\n\n```\n{text}\n```\n")


# -- Step 6: The red-team suite -----------------------------------------------------------
def step_6_red_team() -> list:
    """Run every adversarial case end to end through the guarded workflow and tabulate which
    guardrail fired. A guardrail 'fires' on any non-ALLOW action (BLOCK, ESCALATE, REDACT);
    a REDACT that removes sensitive content and serves the response counts as a successful catch."""
    rows = []
    for name, customer_id, note, msgs, expect in rt.CASES:
        out = asyncio.run(_run_guarded(customer_id, note, msgs))
        fired = out["fired"]
        if expect is None:
            blocked = out["blocked_at"] is not None
            outcome = "PASS" if not blocked else f"FALSE_POSITIVE({out['blocked_at']})"
            fired_at = "served"
        else:
            outcome = "PASS" if expect in fired else f"MISS(fired={sorted(fired)})"
            fired_at = f"{expect}:{fired.get(expect, '-')}"
        rows.append((name, expect or "clean", fired_at, outcome))

    header = f"{'case':<24}{'expected':<18}{'fired':<26}{'outcome':<14}"
    body = [header, "-" * len(header)]
    body += [f"{n:<24}{e:<18}{c:<26}{o:<14}" for n, e, c, o in rows]
    passes = sum(1 for *_, o in rows if o == "PASS")
    body += ["", f"{passes}/{len(rows)} cases handled as expected"]
    table = "\n".join(body)
    print(table)
    _report("oai_step_6_guard_red_team", f"# Step 6 -- red-team results\n\n```\n{table}\n```\n")
    return rows


def main() -> None:
    print("== Step 1: pipeline skeleton ==");     step_1_pipeline_skeleton()
    print("\n== Step 2: input validation ==");    step_2_input_validation()
    print("\n== Step 3: PII redaction ==");        step_3_pii_redaction()
    print("\n== Step 4: injection defence ==");    step_4_injection_defence()
    print("\n== Step 5: output policy ==");        step_5_output_policy()
    print("\n== Step 6: red-team suite ==");       step_6_red_team()


if __name__ == "__main__":
    main()
