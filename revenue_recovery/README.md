# Revenue Recovery Agent

Razorpay Buildathon — Track 03: AI Revenue Recovery.

An agent that looks at a batch of failed payments, decides — under hard
compliance guardrails — whether and how to try to recover each one, phrases
a customer message for whatever action was decided, and reports the actual
measured money recovered with a full per-transaction audit trail.

## Pipeline stages

| Stage | File | What it does |
|---|---|---|
| 1. Detection | `generate_data.py` | Synthesizes `historical_outcomes.csv` (2000 past retries with known outcomes) and `failed_payments.csv` (the current batch of ~120 failed payments to act on). Also defines `true_success_probability`, the hidden ground-truth curve per failure code. |
| 2. Diagnosis | `risk_model.py` | Fits one logistic regression **per failure code** on `[hours_since_failure, retry_count] -> success`, from the historical data. |
| 3. Decision + guardrails | `policy.py` | Hard guardrails run first (no scoring at all if they fire); otherwise scores a fixed menu of actions by expected value and picks the best, subject to soft guardrails. |
| 5. Messaging | `messenger.py` | Turns an already-decided action into a short customer message (English or Hinglish). Never makes contact decisions itself. |
| 6. Orchestration | `main.py` | Runs the whole batch, simulates real outcomes against the hidden ground truth, prints a report, saves the audit trail CSV and a summary chart. |
| — | `test_guardrails.py` | pytest suite that tries to break every guardrail. |
| — | `calibration_check.py` | Train/test split, accuracy + calibration table for the risk model. |

## The math, in plain terms

**Why a separate model per failure code, not one shared model?**
A single logistic regression with one "hours_since_failure" coefficient can
only point in one direction. But `insufficient_funds` gets *more* likely to
succeed the longer you wait (the customer tops up their account), while
`network_timeout` gets *less* likely the longer you wait (it was a one-off
glitch — there's nothing to wait for). One shared coefficient can't be both
positive and negative, so a shared model would learn the wrong direction for
at least one code. Fitting one model per code lets each capture its own
time-trend. `risk_model.py` prints a sanity table (P(success) at 0h/24h/48h
per code) after training specifically to catch this class of bug.

`lost_stolen_card` and `mandate_revoked` are never modeled statistically —
they're hard-coded to `predict_proba() == 0.0` because they're compliance
rules, not probabilities to be estimated.

**Expected value formula**

```
EV(action) = P(success) * amount - cost(action)
```

Candidate actions per transaction: `retry_0h`, `retry_4h`, `retry_24h`
(automatic retries, ~₹1 each), `send_reminder_alt_method` (~₹3, multiplies
the model's probability by 0.8 since it needs the customer to act),
`escalate_human` (~₹45, only considered above ₹5,000, multiplies probability
by 1.1 since a human touch converts a bit better). The policy picks whichever
candidate has the highest EV.

**Guardrail logic**

*Hard* (checked first, return immediately, no EV computed at all):
1. `lost_stolen_card` / `mandate_revoked` → `compliance_stop`, always.
2. `retry_count >= 3` → `give_up_unlikely`, regardless of odds.

*Soft* (constrain which candidates are even considered / accepted):
3. 4-hour cooldown — don't offer `retry_0h` if a prior attempt (`retry_count
   >= 1`) happened less than 4 hours ago.
4. If even the best-scoring candidate has `P(success) < 5%`, override to
   `give_up_unlikely` instead of taking a low-odds action.

Every decision — from every branch — carries a `reasoning` string explaining
why that action was chosen over the alternatives.

**Simulation vs. training data**

The risk model is trained only on noisy historical outcomes. When `main.py`
simulates what *actually* happens to the current batch, it draws from
`generate_data.true_success_probability` directly — the same hidden-truth
function used to label the historical data, but never exposed to the model
itself. This keeps the "real" outcome honest and non-circular.

## How to run

```bash
cd revenue_recovery
python3 -m venv ../venv          # if not already created
source ../venv/bin/activate
pip install -r requirements.txt

python generate_data.py          # Stage 1: creates historical_outcomes.csv, failed_payments.csv
python risk_model.py             # Stage 2: fits models, prints the direction sanity table
python main.py                   # Stage 6: runs the batch, prints report, writes audit trail + chart
python -m pytest test_guardrails.py -v   # guardrail-breaking tests
python calibration_check.py      # accuracy + calibration table
```

Optional: set `ANTHROPIC_API_KEY` before running `main.py` to have
`messenger.py` generate messages dynamically via the `anthropic` SDK
(`claude-sonnet-4-6`) instead of using the offline templates. The pipeline
runs fully offline without it.

## Outputs

- `recovery_audit_trail.csv` — one row per transaction: action, predicted
  probability, expected value, reasoning, whether it recovered, message sent.
  This is the audit deliverable.
- `recovery_summary.png` — revenue at risk vs. recovered, and recovered
  revenue by failure code.

## What to demo

1. **The guardrail tests** (`pytest test_guardrails.py -v`) — each test
   deliberately constructs the most tempting possible scenario (huge amount,
   favorable timing) for a guardrail to fail, and shows it holds anyway. This
   is the evidence that compliance stops aren't just "usually low probability"
   but truly hard-coded, independent of the model.
2. **The calibration table** (`calibration_check.py`) — shows that when the
   model says "50-60% chance," the held-out test transactions in that bucket
   really do succeed at close to that rate. This is the rigor evidence behind
   the expected-value numbers, not just a plausible-looking demo.
3. **The audit trail CSV** — every single decision, including the ones that
   didn't pay off, with a plain-English reasoning string for why it was made.
