# Revenue Recovery Agent

Razorpay Buildathon — Track 03: AI Revenue Recovery.

An agent that looks at a batch of failed payments, decides — under hard
compliance guardrails — whether and how to try to recover each one, phrases
a customer message for whatever action was decided, and reports the actual
measured money recovered with a full per-transaction audit trail.

## Guardrails: bounded and gated by design

This agent's recovery workflow is deliberately **bounded** — it can only ever
pick from a fixed menu of five actions, never invent a new one — and
**gated** by four guardrails that run before any expected-value ranking, so
a bug or a bad prediction in the scoring logic can never override them:
**(1) compliance stop** — `lost/stolen card` and `revoked mandate` failures
are hard-coded to `compliance_stop` and never retried or contacted, under any
circumstances, because retrying a blocked instrument or a revoked mandate is
a compliance violation, not just a low-probability bet; **(2) a max-attempts
ceiling** — `retry_count >= 3` forces `give_up_unlikely` regardless of
predicted odds, so a customer is never contacted indefinitely; **(3) a
4-hour cooldown** — no immediate re-contact is offered within 4 hours of a
prior attempt, so the agent can't spam a customer; **(4) a give-up floor** —
if even the best-scoring candidate action has `P(success) < 5%`, the agent
gives up instead of spending money on a near-hopeless contact. Every one of
these four is exercised by a dedicated pytest in `test_guardrails.py` that
deliberately constructs the most tempting possible scenario (a huge amount,
favorable-looking history) for the guardrail to fail, and confirms it holds
anyway — this is the evidence that the bounds are structural, not just
usually-true heuristics.

## Pipeline stages

| Stage | File | What it does |
|---|---|---|
| 1. Detection | `generate_data.py` | Synthesizes `historical_outcomes.csv` (3000 past retries with known outcomes) and `failed_payments.csv` (the current batch of ~120 failed payments to act on). Also defines `true_success_probability`, the hidden ground-truth curve per failure code. |
| 2. Diagnosis | `risk_model.py` | Fits one logistic regression **per failure code** on `[hours_since_failure, retry_count, is_subscription, payment_method one-hots] -> success`, from the historical data. Also runs a diagnostic comparing accuracy with vs. without the extra features, per code. |
| 3. Decision + guardrails | `policy.py` | Hard guardrails run first (no scoring at all if they fire); otherwise scores a fixed menu of actions by expected value and picks the best, subject to soft guardrails. |
| 5. Messaging | `messenger.py` | Turns an already-decided action into a short customer message (English or Hinglish). Never makes contact decisions itself. |
| 6. Orchestration | `main.py` | Runs the whole batch, simulates real outcomes against the hidden ground truth, prints a report, saves the audit trail CSV and a summary chart. |
| — | `test_guardrails.py` | pytest suite that tries to break every guardrail. |
| — | `calibration_check.py` | 5-fold cross-validation: mean/std accuracy (overall and per code) vs. a naive per-code-average baseline, plus a fold-averaged calibration table. |
| — | `fetch_test_mode_sample.py` | Optional: pulls a handful of real Razorpay Test Mode error responses, if test API keys are provided. |

## The math, in plain terms

**Why a separate model per failure code, not one shared model?**
A single logistic regression with one "hours_since_failure" coefficient can
only point in one direction. But `insufficient_funds` gets *more* likely to
succeed the longer you wait (the customer tops up their account), while
`payment_timed_out` gets *less* likely the longer you wait (it was a one-off
glitch — there's nothing to wait for). One shared coefficient can't be both
positive and negative, so a shared model would learn the wrong direction for
at least one code. Fitting one model per code lets each capture its own
time-trend. `risk_model.py` prints a sanity table (P(success) at 0h/24h/48h
per code) after training specifically to catch this class of bug.

`debit_instrument_blocked` and `mandate_revoked` are never modeled
statistically — they're hard-coded to `predict_proba() == 0.0` because
they're compliance rules, not probabilities to be estimated.

**Do the extra features (`payment_method`, `is_subscription`) actually help?**
Less than you might expect, honestly reported rather than oversold —
`risk_model.py` runs the SAME 5-fold CV `calibration_check.py` uses (not a
single split, which turned out to be misleadingly optimistic here: an
earlier single-split version of this check showed `insufficient_funds`
gaining +0.040 accuracy from the new features, but under proper 5-fold CV
that shrinks to +0.002 — within fold-to-fold noise). At the accuracy@0.5
threshold, none of the six codes show a real gain from the extra features in
cross-validation. That does **not** mean the features carry no signal: for
`payment_timed_out`, the extended model correctly learns P=0.52 for UPI vs.
P=0.07 for netbanking at hour 0 (verified directly against the model, not
inferred) — a real, large, correctly-learned probability difference that
simply doesn't flip enough rows across the 0.5 classification boundary to
move accuracy. Threshold accuracy is a blunt instrument for small-to-moderate
effect sizes; a log-loss or Brier-score comparison would likely show the gain
more clearly, but wasn't implemented here. Codes with too few examples of one
outcome class (e.g. `do_not_honor`) automatically fall back to the original 2
features even when the richer set is requested in some or all folds, rather
than fitting one-hot columns on noise — `risk_model.py`'s
`MIN_MINORITY_CLASS_FOR_EXTRA_FEATURES` gate.

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
1. `debit_instrument_blocked` / `mandate_revoked` → `compliance_stop`, always.
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

## Data provenance

**This is SYNTHETIC data.** No real Razorpay transactions, merchants, or
customers are involved anywhere in this project.

What *is* real: the 6-code failure taxonomy is aligned to Razorpay's actual,
publicly documented payment error reasons
([payments error list](https://razorpay.com/docs/errors/payments/list/),
[card errors](https://razorpay.com/docs/errors/payments/cards/)), cross-checked
by fetching both pages directly:
- `insufficient_funds` is Razorpay's exact real reason string, unchanged.
- `payment_timed_out` and `card_declined` are real Razorpay reason strings we
  renamed our invented codes to, since they're the closest documented
  equivalents (`card_declined` is documented for cards specifically; we use
  it here as our generic-decline analogue across all payment methods, since
  Razorpay doesn't publish a method-agnostic decline reason).
- `debit_instrument_blocked` is a real reason string ("blocked by the issuer
  or by customers themselves") used here as the closest equivalent to a
  lost/stolen card, though it's broader than that specifically.
- `do_not_honor` and `mandate_revoked` are **kept as internal-only labels** —
  neither page documents an exact equivalent (no distinct "do not honor"
  reason, and only mandate *creation*-failure reasons exist, not a
  revocation reason).

What is **not** real: the success-rate curves in
`generate_data.true_success_probability`, and every row in
`historical_outcomes.csv`/`failed_payments.csv`, are simulated. Real
production transaction data was not available for this hackathon, so no
claim is made anywhere in this codebase that the probabilities, recovery
rates, or revenue figures reflect actual Razorpay outcomes — only that the
*category names* are grounded in Razorpay's real documentation.

`fetch_test_mode_sample.py` is an optional, separate script that can pull a
handful of *real* Razorpay Test Mode error responses (not synthetic) if you
provide `RAZORPAY_TEST_KEY_ID`/`RAZORPAY_TEST_KEY_SECRET` test-mode
credentials — see that file's docstring. It has nothing to do with the
synthetic pipeline above; it exists purely so real documented error payloads
can be inspected, if desired, alongside the synthetic ones.

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
python calibration_check.py      # 5-fold CV accuracy + calibration table

# Optional, needs RAZORPAY_TEST_KEY_ID/RAZORPAY_TEST_KEY_SECRET env vars:
python fetch_test_mode_sample.py
```

Optional: set `ANTHROPIC_API_KEY` before running `main.py` to have
`messenger.py` generate messages dynamically via the `anthropic` SDK
(`claude-sonnet-4-6`) instead of using the offline templates. The pipeline
runs fully offline without it.

## Web UI (live demo)

For a live, "agent working" demo instead of reading a console dump, there's a
FastAPI + browser front end that wraps the exact same pipeline modules above
(no logic is duplicated — it imports `generate_data`, `risk_model`, `policy`,
`messenger`, `main` directly):

```bash
cd revenue_recovery/webapp
../../venv/bin/uvicorn server:app --reload --port 8000
```

Then open http://localhost:8000. Click **New Batch** to generate a fresh
random batch, then **Run Agent** to watch it stream through each transaction
live (action, probability, expected value, reasoning, message), with a
summary dashboard, a revenue chart, an audit-trail CSV download, a model
calibration tab, and a live guardrail-test-results tab.

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
2. **The 5-fold cross-validated calibration table** (`calibration_check.py`)
   — mean/std accuracy across 5 folds (not a single lucky/unlucky split),
   compared against a naive "just use the code's historical average" baseline
   so it's clear the model is adding value, plus a fold-averaged calibration
   table showing "50-60% predicted" really does mean "succeeds ~50-60% of the
   time" among held-out transactions.
3. **The feature-value comparison** (`risk_model.py`) — a 5-fold cross-
   validated per-code accuracy comparison of the original 2-feature model
   against the richer model with `payment_method`/`is_subscription` added.
   Reported honestly even when the news is "no measurable gain at the
   accuracy@0.5 threshold, despite real signal underneath" — a good example
   of not oversimplifying evaluation results for a nicer-looking demo.
4. **The audit trail CSV** — every single decision, including the ones that
   didn't pay off, with a plain-English reasoning string for why it was made.
