# Framework decision-matrix template (Chapter 3, customer-care weights)

A ready-to-adapt version of the Chapter 3 scoring rubric and decision matrix.

- `decision_matrix.csv` - the nine decision dimensions (D1-D9), example
  weights tuned for a **customer-care organisation**, and the chapter's
  illustrative 0-5 framework scores. Replace the scores with your own pilot
  results; adjust the weights to your priorities (they must sum to 1.0).
- `score_frameworks.py` - computes weighted scores and prints a ranking:

```
python score_frameworks.py
```

Why these weights? A contact centre runs at **high volume** (D6: cost and
latency per conversation), plugs into **CRM, telephony and ticketing**
(D2), handles **personal data and consumer-protection rules** (D3, D7),
and needs **typed outputs** that a CRM can store (D8). A research or
innovation team would instead up-weight D1, D4 and D5.
