# Complaint-Resolution Evaluation Dataset

Sample dataset for the Chapter 11 harness (customer-care edition). It demonstrates the
schema and exercises every code path; a production dataset is far larger (see below).

## Files

| File | Cases | Purpose |
|------|-------|---------|
| `core_set.jsonl` | 5 | One case per decision type (APPROVE, PARTIAL, DENY, ESCALATE) from the NovaMart customers used across the course |
| `safety_set.jsonl` | 5 | 3 real hazards (smoke, electric shock, swollen battery) + 2 look-alikes ("hot like fire", "phone gets warm") |
| `adversarial_set.jsonl` | 5 | Direct and role-play prompt injection, Hindi-English code-mixing, an all-caps threat, an indirect injection asking for payment to a third-party UPI handle |
| `disparate_impact_set.jsonl` | 6 | The same mixer-grinder complaint from Hindi, Tamil and English speakers (2 each) |
| `calibration_set.jsonl` | 5 | Human-labelled grounding examples for calibrating the LLM judge |

## Schema

Each case line conforms to `src.models.ComplaintCase`:

- `case_id`: unique, segment-prefixed (`core-001`, `saf-001`, `adv-001`, `di-001`)
- `segment`: `core`, `safety`, `adversarial`, `disparate_impact` or `regression`
- `customer`: name, customer ID, tier, preferred language, region, the complaint text
- `orders`: order references (ID, product, value in INR, delivery date)
- `expected`: gold decision, safety-escalation flag, required tools, rationale
- `metadata`: provenance (source, created_at, labelers, kappa)

## Production dataset construction (suggested)

- **Core set: ~1,000 cases** stratified by decision (250 each), from closed tickets with a
  confirmed outcome, anonymised
- **Safety set: ~200 cases**: 100 confirmed hazards from the product-safety log, 100 look-alikes
- **Adversarial set: ~150 cases** from a red team: injections, code-mixed language,
  voice-transcription noise, social-engineering attempts
- **Disparate-impact set: ~500 cases** balanced across languages, regions and customer tiers
- **Live regression set: ~200 cases** sampled from production every quarter

## Labelling protocol

- Two customer-care quality leads label each case independently
- A senior reviewer adjudicates disagreements
- Cohen's kappa is measured on every batch; minimum 0.7 for production use.
  (The 5-item calibration sample here has kappa 0.44 on purpose: two items are disputed,
  which is exactly the signal that the rubric needs work.)
