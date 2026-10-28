"""Self-Refine (Madaan et al., 2023) - customer-care edition.

One LLM plays three roles in a loop:
  1. generator - drafts a resolution letter for an escalated complaint,
  2. reviewer  - gives specific, actionable feedback,
  3. refiner   - rewrites the letter to address the feedback,
until the reviewer answers NO_FURTHER_FEEDBACK or MAX_ITERATIONS is reached.

Finance edition: the same loop refined an equity investment thesis.
"""
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = os.getenv("CC_ANTHROPIC_MODEL", "claude-opus-5-5")
MAX_ITERATIONS = 3


def ask_claude(prompt: str, max_tokens: int = 16000) -> str:
    """One Claude call. fallbacks="default" re-runs a request that a safety classifier declines
    on another Claude model; we still check stop_reason before reading the text."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request: {response.stop_details}")
    return "".join(block.text for block in response.content if block.type == "text")


def generate_initial_letter(case: str) -> str:
    prompt = f"""You are a senior customer-care manager at NovaMart, an Indian electronics retailer.
Draft a resolution letter to the customer based on the following case data:

{case}

The letter must cover: an apology that acknowledges the specific failures, the resolution and its
timeline, the goodwill gesture, and what NovaMart is changing so it does not happen again.
Use only facts present in the case data [Case data]; flag anything else as [unverified].
Write 2-3 sentences per section. Be concise and warm, not defensive."""
    return ask_claude(prompt)


def get_feedback(letter: str, case: str, history_feedback: str) -> str:
    prompt = f"""You are a critical reviewer on NovaMart's escalation desk (quality, legal and brand).
Review the resolution letter below and give specific, actionable feedback. Check:
- accuracy: every fact, amount and date matches the case data; nothing is invented;
- commitments: each has a concrete date and is within policy and approval limits;
- tone: empathetic, no blame on the customer, no defensive language, no legal admissions;
- completeness: apology, resolution and timeline, goodwill, prevention.
Ground your feedback only in the case data [Case data]; flag any external claim as [unverified].
If no issue would block sending this letter, respond with exactly: NO_FURTHER_FEEDBACK
Be concise.

Case data:
{case}

Previous letters and feedback:
{history_feedback}

Letter to review:
{letter}

Feedback:"""
    return ask_claude(prompt)


def refine_letter(letter: str, feedback: str, case: str, history_feedback: str) -> str:
    prompt = f"""You are a senior customer-care manager at NovaMart.
Revise the resolution letter below so it addresses every point in the reviewer feedback.
Use only facts present in the case data [Case data]; flag anything else as [unverified].
Keep the same four sections: apology, resolution and timeline, goodwill, prevention.
Write 2-3 sentences per section. Be concise.

Case data:
{case}

Previous letters and feedback:
{history_feedback}

Letter to revise:
{letter}

Reviewer feedback:
{feedback}

Revised letter:"""
    return ask_claude(prompt)


def self_refine_letter(case: str) -> tuple[str, str]:
    print("[STEP 1] Generating the initial letter ...")
    letter = generate_initial_letter(case)
    history_feedback = ""
    print(f"Initial letter:\n{letter}\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"[STEP 2] Iteration {iteration} - getting feedback ...")
        feedback = get_feedback(letter, case, history_feedback)
        print(f"Feedback:\n{feedback}\n")
        if feedback.strip().startswith("NO_FURTHER_FEEDBACK"):
            print("Stopping: the reviewer reports no further improvements needed.")
            break
        history_feedback += f"Iteration: {iteration}\nLetter: {letter}\nFeedback: {feedback}\n\n"
        print(f"[STEP 3] Iteration {iteration} - refining the letter ...")
        letter = refine_letter(letter, feedback, case, history_feedback)
        print(f"Refined letter:\n{letter}\n")
    return letter, history_feedback


case = """
----- CASE C1001 / ORDER ORD-77812 -----
Customer: Priya Sharma, Platinum member since 2019 (lifetime spend Rs 4,85,000; 38 orders)
Product: FrostLine Refrigerator 340L, Rs 62,990, delivered 18 September 2026
Issue: stopped cooling on 9 October; repairs on 12 October and 19 October failed
Root cause (internal): compressor relay part R-22 failures in July-August builds; part back-ordered until 15 November
Customer loss claimed: food worth about Rs 4,000 discarded; elderly parents at home
Policy: repeat-failure rule KB-102 -> replacement of the same model; Platinum priority dispatch within 72 hours
Goodwill authority: agent up to Rs 1,000; supervisor up to Rs 5,000
Approved by supervisor Kavita Rao: Rs 3,000 wallet credit; installation charge waived
Replacement reference: RPL-90318; same model in stock at the Pune warehouse; delivery and installation by Thursday 29 October
Customer signals: public Twitter post (1.2k reposts); consumer-court notice threatened if not replaced by 29 October
"""

if __name__ == "__main__":
    # Remark: the prompt grows with every iteration (history_feedback), and so does the cost.
    final_letter, history = self_refine_letter(case)
    print("FINAL LETTER:\n", final_letter)
