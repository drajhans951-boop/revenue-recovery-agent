"""
Stage 1 - Detection: synthetic data generation.

Produces two CSVs:
  historical_outcomes.csv - past retry attempts with known outcomes, for training.
  failed_payments.csv     - the CURRENT batch of failed payments to process.

`true_success_probability` is the single "hidden truth" generator used both to
label historical rows AND (imported directly, later, by main.py) to simulate
what actually happens when the agent acts on the current batch. The risk model
in risk_model.py never sees this function - it only sees noisy Bernoulli draws
from it, exactly like a real fraud/recovery model only sees realized outcomes.

Data provenance
----------------
This is SYNTHETIC data - no real Razorpay transactions were used. The 6-code
failure taxonomy below is aligned to Razorpay's real, publicly documented
error reasons where a genuine equivalent exists (cross-referenced against
https://razorpay.com/docs/errors/payments/list/ and
https://razorpay.com/docs/errors/payments/cards/):
  - insufficient_funds        -> exact match to Razorpay's real reason string.
  - payment_timed_out         -> renamed from an invented "network_timeout";
                                  matches Razorpay's documented payment_timed_out.
  - card_declined             -> renamed from an invented "bank_declined_generic";
                                  Razorpay documents card_declined specifically for
                                  cards, we use it here as our closest generic-decline
                                  analogue across all payment methods since Razorpay
                                  does not publish a method-agnostic decline reason.
  - do_not_honor               -> KEPT as an internal-only label. Neither Razorpay
                                  page documents a "do not honor" style reason; this
                                  approximates the classic ISO 8583 code-05 bank
                                  response, which Razorpay does not expose distinctly.
  - debit_instrument_blocked  -> renamed from an invented "lost_stolen_card";
                                  Razorpay's debit_instrument_blocked ("blocked by
                                  the issuer or by customers themselves") is the
                                  closest real equivalent, though it is broader than
                                  lost/stolen specifically and documented for cards
                                  while we apply it across all payment methods.
  - mandate_revoked            -> KEPT as an internal-only label. Razorpay only
                                  documents mandate *creation*-failure reasons
                                  (mandate_creation_declined/failed/expired), not a
                                  revocation/cancellation reason.
The success-rate CURVES and historical OUTCOMES below are simulated, not drawn
from real transaction history - real production data isn't available for this
hackathon. Do not present this dataset as real transaction data anywhere.
"""
import numpy as np
import pandas as pd

FAILURE_CODES = [
    "insufficient_funds",
    "payment_timed_out",
    "card_declined",
    "do_not_honor",
    "debit_instrument_blocked",
    "mandate_revoked",
]

COMPLIANCE_HARD_STOP_CODES = {"debit_instrument_blocked", "mandate_revoked"}

RANDOM_SEED = 42

PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet"]


def true_success_probability(failure_code: str, hours_since_failure: float, retry_count: int,
                              payment_method: str = "upi", is_subscription: bool = False) -> float:
    """The real-world (hidden) probability that a retry/contact succeeds.

    Each failure code gets a genuinely different time-dynamic - this is the
    whole point of fitting a separate model per code in risk_model.py. A
    retry "fatigue" factor is applied on top: every extra attempt makes the
    customer somewhat less likely to convert, for every code.

    payment_method/is_subscription deliberately affect only SOME codes below,
    and by design amounts of zero for the others (card_declined, do_not_honor)
    - this lets risk_model.py's feature-value comparison honestly report "no
    improvement" for those codes instead of us forcing signal that isn't real.
    """
    h = max(0.0, float(hours_since_failure))

    if failure_code in COMPLIANCE_HARD_STOP_CODES:
        # Compliance hard-stop: never recoverable, independent of anything else.
        return 0.0
    elif failure_code == "insufficient_funds":
        # Rises with time (people get paid / top up), plateaus after ~24h.
        p = 0.15 + 0.60 * (1 - np.exp(-h / 10.0))
        # Autopay/mandate-linked failures on funds tend to be stickier: it's
        # the same low-balance account being retried, not a fresh attempt.
        if is_subscription:
            p -= 0.22
    elif failure_code == "payment_timed_out":
        # High immediately (transient glitch), decays fast - no reason to wait.
        p = 0.15 + 0.55 * np.exp(-h / 6.0)
        # Channel matters for a timeout: UPI retries are near-instant
        # app-to-app, netbanking retries re-enter a bank gateway session
        # that may itself be slow/expired again.
        if payment_method == "upi":
            p += 0.22
        elif payment_method == "netbanking":
            p -= 0.15
    elif failure_code == "card_declined":
        # Low, roughly flat with a slight upward drift. No dependence on
        # payment_method/is_subscription - a generic decline is noise here.
        p = 0.10 + 0.002 * h
        p = min(p, 0.18)
    elif failure_code == "do_not_honor":
        # Very low across the board, barely recoverable. No dependence on
        # payment_method/is_subscription - deliberately noise-only.
        p = 0.03 + 0.0005 * h
        p = min(p, 0.06)
    else:
        raise ValueError(f"Unknown failure_code: {failure_code}")

    fatigue = 0.8 ** max(0, int(retry_count))
    p = p * fatigue
    return float(np.clip(p, 0.0, 1.0))


def generate_historical_outcomes(n_rows: int = 3000, rng: np.random.Generator = None) -> pd.DataFrame:
    """Past retry attempts with known outcomes, used to train the risk model.

    n_rows=3000 (up from an original 2000): at 2000 rows, low-base-rate codes
    like card_declined (~15-20% of rows, success rate under 10%) had too few
    examples per retry_count bin for the fatigue effect's small absolute
    swing to reliably beat sampling noise - risk_model.py's own sanity check
    caught the fitted retry_count coefficient landing with the wrong sign on
    a low-luck seed. More rows reduces that estimation variance directly.
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)

    # Skew towards the common, actionable codes; compliance codes are rarer
    # in history but still present (they show up in real transaction logs).
    code_weights = {
        "insufficient_funds": 0.30,
        "payment_timed_out": 0.25,
        "card_declined": 0.20,
        "do_not_honor": 0.12,
        "debit_instrument_blocked": 0.07,
        "mandate_revoked": 0.06,
    }
    codes = rng.choice(
        list(code_weights.keys()), size=n_rows, p=list(code_weights.values())
    )

    hours_since_failure = rng.uniform(0, 72, size=n_rows)
    retry_count = rng.integers(0, 4, size=n_rows)  # 0..3
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows)
    is_subscription = rng.choice([True, False], size=n_rows, p=[0.35, 0.65])

    success = np.empty(n_rows, dtype=int)
    for i in range(n_rows):
        p = true_success_probability(codes[i], hours_since_failure[i], retry_count[i],
                                      payment_method[i], is_subscription[i])
        success[i] = rng.binomial(1, p)

    return pd.DataFrame(
        {
            "failure_code": codes,
            "hours_since_failure": np.round(hours_since_failure, 2),
            "retry_count": retry_count,
            "payment_method": payment_method,
            "is_subscription": is_subscription,
            "success": success,
        }
    )


def generate_failed_payments(n_rows: int = 120, rng: np.random.Generator = None) -> pd.DataFrame:
    """The CURRENT batch of failed payments the agent must act on."""
    rng = rng or np.random.default_rng(RANDOM_SEED + 1)

    # All 6 codes appear so guardrails actually get exercised in the real run.
    code_weights = {
        "insufficient_funds": 0.28,
        "payment_timed_out": 0.22,
        "card_declined": 0.18,
        "do_not_honor": 0.14,
        "debit_instrument_blocked": 0.09,
        "mandate_revoked": 0.09,
    }
    codes = rng.choice(
        list(code_weights.keys()), size=n_rows, p=list(code_weights.values())
    )

    # Lognormal-ish amounts, clipped to a realistic fintech range.
    amount = np.round(np.clip(rng.lognormal(mean=7.5, sigma=1.0, size=n_rows), 100, 50000), 2)
    payment_method = rng.choice(PAYMENT_METHODS, size=n_rows)
    is_subscription = rng.choice([True, False], size=n_rows, p=[0.35, 0.65])
    preferred_lang = rng.choice(["en", "hi"], size=n_rows, p=[0.6, 0.4])

    return pd.DataFrame(
        {
            "txn_id": [f"TXN{100000 + i}" for i in range(n_rows)],
            "customer_id": [f"CUST{rng.integers(1000, 9999)}" for _ in range(n_rows)],
            "amount": amount,
            "failure_code": codes,
            "payment_method": payment_method,
            "is_subscription": is_subscription,
            "hours_since_failure": 0.0,
            "retry_count": 0,
            "preferred_lang": preferred_lang,
        }
    )


def main():
    rng_hist = np.random.default_rng(RANDOM_SEED)
    rng_batch = np.random.default_rng(RANDOM_SEED + 1)

    historical = generate_historical_outcomes(3000, rng_hist)
    batch = generate_failed_payments(120, rng_batch)

    historical.to_csv("historical_outcomes.csv", index=False)
    batch.to_csv("failed_payments.csv", index=False)

    print(f"Wrote historical_outcomes.csv: {len(historical)} rows")
    print(historical["failure_code"].value_counts())
    print(f"\nOverall historical success rate: {historical['success'].mean():.3f}")

    print(f"\nWrote failed_payments.csv: {len(batch)} rows")
    print(batch["failure_code"].value_counts())
    print(f"\nTotal revenue at risk in batch: Rs.{batch['amount'].sum():,.2f}")


if __name__ == "__main__":
    main()
