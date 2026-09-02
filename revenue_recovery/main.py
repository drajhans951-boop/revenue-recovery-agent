"""
Stage 6 - Orchestration + reporting.

Runs the full pipeline over the current batch of failed payments:
  1. Load both CSVs, fit the risk model on historical_outcomes.csv.
  2. For each transaction: policy.decide() -> messenger.generate_message().
  3. Simulate the REAL outcome using generate_data.true_success_probability
     directly (the "hidden truth" generator) - NOT the fitted model's own
     prediction, which would be circular.
  4. Aggregate and report, honestly, including the failures.

Outputs:
  - console report
  - recovery_audit_trail.csv (one row per transaction - the audit deliverable)
  - recovery_summary.png (revenue at risk vs recovered, and by failure code)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_data import true_success_probability
from risk_model import fit_risk_models
from policy import decide, ALT_METHOD_MULTIPLIER, ESCALATE_MULTIPLIER
from messenger import generate_message

SIMULATION_SEED = 2026

# Maps a chosen action to the (hours_since_failure, retry_count) the customer
# is effectively being asked to convert at - mirrors the assumptions baked
# into policy._score_candidate, so the simulated outcome is consistent with
# what the decision claimed to expect.
ACTION_TIMING = {
    "retry_0h": 0,
    "retry_4h": 4,
    "retry_24h": 24,
}


def simulate_actual_outcome(action: str, failure_code: str, retry_count: int, rng: np.random.Generator,
                             payment_method: str = "upi", is_subscription: bool = False) -> bool:
    """Draw the REAL outcome from the hidden-truth generator, independent of
    what the risk model predicted. Returns True if the action recovers the
    payment."""
    if action in ("compliance_stop", "give_up_unlikely"):
        return False

    if action in ACTION_TIMING:
        hours = ACTION_TIMING[action]
        p_true = true_success_probability(failure_code, hours, retry_count, payment_method, is_subscription)
    elif action == "send_reminder_alt_method":
        p_true = true_success_probability(failure_code, 0, retry_count, payment_method,
                                           is_subscription) * ALT_METHOD_MULTIPLIER
    elif action == "escalate_human":
        p_true = min(1.0, true_success_probability(failure_code, 0, retry_count, payment_method,
                                                     is_subscription) * ESCALATE_MULTIPLIER)
    else:
        raise ValueError(f"Unknown action: {action}")

    return bool(rng.binomial(1, np.clip(p_true, 0.0, 1.0)))


def run_pipeline():
    historical = pd.read_csv("historical_outcomes.csv")
    batch = pd.read_csv("failed_payments.csv")

    models = fit_risk_models(historical)
    rng = np.random.default_rng(SIMULATION_SEED)

    rows = []
    for _, txn in batch.iterrows():
        txn_dict = txn.to_dict()
        d = decide(txn_dict, models)
        message = generate_message(d.action, txn["preferred_lang"], txn["amount"], txn["failure_code"])
        recovered = simulate_actual_outcome(d.action, txn["failure_code"], int(txn["retry_count"]), rng,
                                             txn["payment_method"], bool(txn["is_subscription"]))

        rows.append(
            {
                "txn_id": d.txn_id,
                "customer_id": txn["customer_id"],
                "amount": txn["amount"],
                "failure_code": txn["failure_code"],
                "action": d.action,
                "predicted_prob": round(d.predicted_prob, 4),
                "expected_value": round(d.expected_value, 2),
                "reasoning": d.reasoning,
                "recovered": recovered,
                "revenue_recovered": txn["amount"] if recovered else 0.0,
                "message": message,
            }
        )

    return pd.DataFrame(rows)


def print_report(audit: pd.DataFrame):
    n = len(audit)
    total_at_risk = audit["amount"].sum()
    total_recovered = audit["revenue_recovered"].sum()
    recovery_rate = total_recovered / total_at_risk if total_at_risk else 0.0

    print("=" * 72)
    print("REVENUE RECOVERY AGENT - BATCH REPORT")
    print("=" * 72)
    print(f"Transactions processed:     {n}")
    print(f"Total revenue at risk:      Rs.{total_at_risk:,.2f}")
    print(f"Total revenue recovered:    Rs.{total_recovered:,.2f}")
    print(f"Recovery rate:              {recovery_rate:.2%}")

    print("\nAction breakdown:")
    print(audit["action"].value_counts().to_string())

    hard_stops = (audit["action"] == "compliance_stop").sum()
    soft_giveups = (audit["action"] == "give_up_unlikely").sum()
    print(f"\nCompliance hard-stops (blocked instrument, revoked mandate): {hard_stops}")
    print(f"Soft give-ups (max retries reached or odds too low):       {soft_giveups}")

    print("\n" + "-" * 72)
    print("SAMPLE DECISIONS")
    print("-" * 72)
    sample = audit.sample(n=min(6, n), random_state=7)
    for _, r in sample.iterrows():
        print(f"\n[{r['txn_id']}] amount=Rs.{r['amount']:.2f} failure_code={r['failure_code']}")
        print(f"  action={r['action']}  P={r['predicted_prob']:.3f}  EV=Rs.{r['expected_value']:.2f}  recovered={r['recovered']}")
        print(f"  reasoning: {r['reasoning']}")
        message = r["message"] if isinstance(r["message"], str) else "(no message sent)"
        print(f"  message: {message}")
    print("=" * 72)


def save_chart(audit: pd.DataFrame, path: str = "recovery_summary.png"):
    total_at_risk = audit["amount"].sum()
    total_recovered = audit["revenue_recovered"].sum()

    by_code = audit.groupby("failure_code")["revenue_recovered"].sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].bar(["Revenue at risk", "Revenue recovered"], [total_at_risk, total_recovered],
                color=["#c0392b", "#27ae60"])
    axes[0].set_title("Revenue at risk vs recovered")
    axes[0].set_ylabel("Rupees")
    for i, v in enumerate([total_at_risk, total_recovered]):
        axes[0].text(i, v, f"Rs.{v:,.0f}", ha="center", va="bottom")

    axes[1].bar(by_code.index, by_code.values, color="#2980b9")
    axes[1].set_title("Revenue recovered by failure code")
    axes[1].set_ylabel("Rupees")
    axes[1].tick_params(axis="x", rotation=40)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nSaved chart to {path}")


def main():
    audit = run_pipeline()
    print_report(audit)
    audit.to_csv("recovery_audit_trail.csv", index=False)
    print("\nSaved audit trail to recovery_audit_trail.csv")
    save_chart(audit)


if __name__ == "__main__":
    main()
