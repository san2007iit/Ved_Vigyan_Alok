# Chapter 8 - Multi-Agent Escalation Committee (customer-care edition)

Two multi-agent architectures built with [LangGraph](https://langchain-ai.github.io/langgraph/)
that compose into one pipeline, plus a replay harness that scores the committee on
past cases:

1. **Hierarchical Escalation Committee** (`escalation_committee.py`). Four specialist
   analysts run in parallel, a Case Manager synthesises a recommendation, and a
   Compliance Officer applies a hard rule plus an LLM check.
2. **Adversarial Debate** on top of the committee's output (`adversarial_debate.py`).
   A Customer Advocate and a Cost Controller argue on **different model families**
   (Anthropic and OpenAI) on purpose; a Devil's Advocate and a Judge settle it.
3. **Historical case replay** (`backtest.py`). Runs the committee on eight closed cases
   in date order and reports credits paid, expected CLV saved, net value, ROI, hit
   rate, max drawdown and agreement with a perfectly informed "oracle".

The question every run answers: *for this escalated case, on top of the policy remedy,
should NovaMart COMPENSATE (and with how large a goodwill credit), handle it STANDARD
(policy only), or DECLINE the claim and investigate?*

## Finance edition -> customer-care edition

| Finance (investment committee) | Customer care (escalation committee) |
|---|---|
| Ticker (TSLA) | Escalated case (CASE-9031 Priya, CASE-9044 Vikram, CASE-9050 Rahul) |
| Fundamentals analyst (P/E, ROE, margins) | **Customer value analyst** (CLV, tenure, lifetime spend, margin) |
| Technicals analyst (SMA, RSI, volatility) | **Service trend analyst** (contact spike vs normal, CSAT momentum, days open, failed fixes) |
| Sentiment analyst (news headlines) | **Voice of customer analyst** (the customer's own messages, public vs private channels) |
| Macro analyst (VIX, 10-year yield) | **Operations analyst** (festive backlog, spare parts, goodwill budget, competitor offers) |
| Portfolio Manager: `ACTION=BUY/HOLD/SELL; SIZE_PCT` of NAV | **Case Manager**: `ACTION=COMPENSATE/STANDARD/DECLINE; CREDIT_PCT` of order value |
| Risk Officer: veto if 30-day volatility > 60%, cap 3% per name | **Compliance Officer**: veto if abuse-risk score > 0.60, cap 10% credit per case (policy KB-103) |
| Bull vs Bear | **Customer Advocate vs Cost Controller** |
| Walk-forward backtest: Sharpe, drawdown, hit rate | **Case replay**: ROI on goodwill, drawdown of cumulative net value, hit rate |

## Architecture

```
                 +--> customer_value ----+
                 +--> service_trend -----+
   START --------+--> voice_of_customer -+--> case_manager --> compliance_officer --> END
                 +--> operations --------+

   committee result --> advocate (Claude)   --+
                    --> controller (OpenAI) --+--> devil's advocate --> judge --> verdict
```

* Every analyst writes its own field of the shared `CommitteeState`, so the parallel
  fan-out from `START` needs no reducer.
* Numbers (contact spike, CSAT momentum, abuse-risk score) are computed in Python before
  the LLM sees them, so they are auditable.
* The abuse-risk veto runs **before** any LLM call: a rule nobody can talk the model out of.

## Models

| Role | Model (env override) |
|---|---|
| Case Manager, Compliance Officer, Customer Advocate | `claude-opus-5-5` (`CC_OPUS_MODEL`) |
| Value, Service, Operations analysts, Devil's Advocate, Judge | `claude-sonnet-5-5` (`CC_SONNET_MODEL`) |
| Voice of customer (high volume) | `claude-haiku-5-5` (`CC_HAIKU_MODEL`) |
| Cost Controller | `gpt-4o` (`CC_OPENAI_MODEL`) |

No `temperature` is set anywhere: recent Claude models reject it. Opus and Sonnet calls
pass `betas=["server-side-fallback-2026-07-01"]` and `fallbacks="default"`, so a
request declined by a safety classifier is re-run on another Claude model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # add ANTHROPIC_API_KEY (and OPENAI_API_KEY for the debate)
```

## Running

```bash
python escalation_committee.py              # CASE-9031: Priya, Platinum, failed repairs
python escalation_committee.py CASE-9044    # Vikram: return-abuse signals -> compliance veto
python adversarial_debate.py CASE-9050      # Rahul: late phone, polite -> does the debate agree?
python backtest.py                          # replay on H-7001 ... H-7008
```

## What the replay does better than the finance backtest

The finance backtest could not be point-in-time: `yfinance` returns *today's*
fundamentals and news even for a 2024 decision date. Here every case is a snapshot of
what was known when it was decided, including the operations picture that month, and
the churn estimates used for scoring are never shown to the committee.

It is still an estimate. `churn_if_standard` and `churn_if_compensated` come from an
uplift model (past A/B tests of goodwill offers), not from observing both outcomes.
The real proof is a randomised hold-out: give standard handling to a random slice of
eligible cases and compare retention.

## Exercises

1. Run CASE-9031 three times. Does the Case Manager's `CREDIT_PCT` move? Why might you
   want the analysts' numbers but the Case Manager's judgement?
2. Raise the compliance cap from 10% to 15% and re-run the replay. Does ROI improve or
   fall? (Hint: credits above 5% buy no extra retention in the uplift model.)
3. Make the Cost Controller run on Claude too. Does the Judge still disagree with the
   committee as often?
4. Add a fifth analyst, `warranty_policy`, that reports which remedies the policy allows.
   Which edges do you add to the graph?

## Disclaimer

Educational use only. All customers, cases and numbers are fictional. In a real
deployment, a human signs off every goodwill commitment above the agent's authority,
and value-based treatment must be checked for fairness (Chapter 11).
