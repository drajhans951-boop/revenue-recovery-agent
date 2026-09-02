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
"""
import numpy as np
import pandas as pd

FAILURE_CODES = [
    "insufficient_funds",
    "network_timeout",
    "bank_declined_generic",
    "do_not_honor",
    "lost_stolen_card",
    "mandate_revoked",
]

COMPLIANCE_HARD_STOP_CODES = {"lost_stolen_card", "mandate_revoked"}

RANDOM_SEED = 42

PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet"]


def true_success_probability(failure_code: str, hours_since_failure: float, retry_count: int) -> float:
    """The real-world (hidden) probability that a retry/contact succeeds.

    Each failure code gets a genuinely different time-dynamic - this is the
    whole point of fitting a separate model per code in risk_model.py. A
    retry "fatigue" factor is applied on top: every extra attempt makes the
    customer somewhat less likely to convert, for every code.
    """
    h = max(0.0, float(hours_since_failure))

    if failure_code in COMPLIANCE_HARD_STOP_CODES:
        # Compliance hard-stop: never recoverable, independent of anything else.
        return 0.0
    elif failure_code == "insufficient_funds":
        # Rises with time (people get paid / top up), plateaus after ~24h.
        p = 0.15 + 0.60 * (1 - np.exp(-h / 10.0))
    elif failure_code == "network_timeout":
        # High immediately (transient glitch), decays fast - no reason to wait.
        p = 0.15 + 0.55 * np.exp(-h / 6.0)
    elif failure_code == "bank_declined_generic":
        # Low, roughly flat with a slight upward drift.
        p = 0.10 + 0.002 * h
        p = min(p, 0.18)
    elif failure_code == "do_not_honor":
        # Very low across the board, barely recoverable.
        p = 0.03 + 0.0005 * h
        p = min(p, 0.06)
    else:
        raise ValueError(f"Unknown failure_code: {failure_code}")

    fatigue = 0.8 ** max(0, int(retry_count))
    p = p * fatigue
    return float(np.clip(p, 0.0, 1.0))


def generate_historical_outcomes(n_rows: int = 2000, rng: np.random.Generator = None) -> pd.DataFrame:
    """Past retry attempts with known outcomes, used to train the risk model."""
    rng = rng or np.random.default_rng(RANDOM_SEED)

    # Skew towards the common, actionable codes; compliance codes are rarer
    # in history but still present (they show up in real transaction logs).
    code_weights = {
        "insufficient_funds": 0.30,
        "network_timeout": 0.25,
        "bank_declined_generic": 0.20,
        "do_not_honor": 0.12,
        "lost_stolen_card": 0.07,
        "mandate_revoked": 0.06,
    }
    codes = rng.choice(
        list(code_weights.keys()), size=n_rows, p=list(code_weights.values())
    )

    hours_since_failure = rng.uniform(0, 72, size=n_rows)
    retry_count = rng.integers(0, 4, size=n_rows)  # 0..3

    success = np.empty(n_rows, dtype=int)
    for i in range(n_rows):
        p = true_success_probability(codes[i], hours_since_failure[i], retry_count[i])
        success[i] = rng.binomial(1, p)

    return pd.DataFrame(
        {
            "failure_code": codes,
            "hours_since_failure": np.round(hours_since_failure, 2),
            "retry_count": retry_count,
            "success": success,
        }
    )


def generate_failed_payments(n_rows: int = 120, rng: np.random.Generator = None) -> pd.DataFrame:
    """The CURRENT batch of failed payments the agent must act on."""
    rng = rng or np.random.default_rng(RANDOM_SEED + 1)

    # All 6 codes appear so guardrails actually get exercised in the real run.
    code_weights = {
        "insufficient_funds": 0.28,
        "network_timeout": 0.22,
        "bank_declined_generic": 0.18,
        "do_not_honor": 0.14,
        "lost_stolen_card": 0.09,
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

    historical = generate_historical_outcomes(2000, rng_hist)
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
