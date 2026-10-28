---
name: staffing-evaluator
description: Score a partial contact-centre staffing trajectory. Use when asked to
  evaluate how close a trajectory is to the target wait time at acceptable cost.
---

Score the trajectory from 1 to 10 using these criteria:

- 6-10: progress toward the target wait time exceeds the elapsed share of the plan,
  extra cost below Rs 8k per day for this action, and the extra capacity is spread
  across the queues that actually have long waits.
- 3-5: on pace, but extra cost between Rs 8k and Rs 15k per day, or all capacity
  concentrated in one queue (agent burnout and single-point-of-failure risk).
- 1-2: behind schedule, or extra cost above Rs 15k per day for a single action.

Return a single integer score and a one-sentence justification.
