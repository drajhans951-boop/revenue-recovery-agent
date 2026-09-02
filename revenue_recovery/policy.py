"""
Stage 3 - Decision engine + guardrails.

Decision order for every transaction:
  1. HARD guardrails (no EV computation happens at all if one fires).
  2. Build the candidate action menu, filtered by SOFT guardrails.
  3. Score each remaining candidate by expected value, pick the best.
  4. Low-odds override: if even the best candidate is below MIN_PROB_TO_ACT,
     give up instead of taking a low-odds action.

Every branch returns a Decision with a populated `reasoning` string.
"""
from dataclasses import dataclass

from risk_model import predict_proba

COMPLIANCE_HARD_STOP_CODES = {"lost_stolen_card", "mandate_revoked"}
MAX_RETRY_COUNT = 3
COOLDOWN_HOURS = 4
MIN_PROB_TO_ACT = 0.05
ESCALATE_AMOUNT_THRESHOLD = 5000.0

ALT_METHOD_MULTIPLIER = 0.8   # customer must act themselves -> converts worse
ESCALATE_MULTIPLIER = 1.1     # human touch converts a bit better

# Costs in rupees. Cheap automatic actions are preferred unless a pricier
# action has a meaningfully better conversion probability.
COSTS = {
    "retry_0h": 1.0,
    "retry_4h": 1.0,
    "retry_24h": 1.0,
    "send_reminder_alt_method": 3.0,
    "escalate_human": 45.0,
}


@dataclass
class Decision:
    txn_id: str
    action: str
    predicted_prob: float
    expected_value: float
    reasoning: str


def _candidate_actions(hours_since_failure: float, retry_count: int, amount: float):
    """The fixed menu, filtered by soft guardrails."""
    actions = ["retry_0h", "retry_4h", "retry_24h", "send_reminder_alt_method"]

    # Soft guardrail: 4h cooldown between contact attempts. Only relevant once
    # a prior attempt has actually been made (retry_count >= 1); a brand-new
    # failure (retry_count == 0) has had no prior contact to cool down from.
    if retry_count >= 1 and hours_since_failure < COOLDOWN_HOURS:
        actions.remove("retry_0h")

    if amount > ESCALATE_AMOUNT_THRESHOLD:
        actions.append("escalate_human")

    return actions


def _score_candidate(action: str, models: dict, failure_code: str, hours_since_failure: float,
                      retry_count: int, amount: float):
    """Return (predicted_prob, expected_value) for one candidate action."""
    if action == "retry_0h":
        prob = predict_proba(models, failure_code, 0, retry_count)
    elif action == "retry_4h":
        prob = predict_proba(models, failure_code, 4, retry_count)
    elif action == "retry_24h":
        prob = predict_proba(models, failure_code, 24, retry_count)
    elif action == "send_reminder_alt_method":
        prob = predict_proba(models, failure_code, hours_since_failure, retry_count) * ALT_METHOD_MULTIPLIER
    elif action == "escalate_human":
        prob = min(1.0, predict_proba(models, failure_code, hours_since_failure, retry_count) * ESCALATE_MULTIPLIER)
    else:
        raise ValueError(f"Unknown action: {action}")

    ev = prob * amount - COSTS[action]
    return prob, ev


def decide(txn: dict, models: dict) -> Decision:
    """txn is expected to have: txn_id, amount, failure_code,
    hours_since_failure, retry_count."""
    txn_id = txn["txn_id"]
    amount = float(txn["amount"])
    failure_code = txn["failure_code"]
    hours_since_failure = float(txn["hours_since_failure"])
    retry_count = int(txn["retry_count"])

    # --- HARD GUARDRAILS: evaluated first, return immediately, no EV at all ---
    if failure_code in COMPLIANCE_HARD_STOP_CODES:
        return Decision(
            txn_id=txn_id,
            action="compliance_stop",
            predicted_prob=0.0,
            expected_value=0.0,
            reasoning=(
                f"Hard compliance guardrail: failure_code='{failure_code}' is a permanent "
                "stop condition (lost/stolen card or revoked mandate). No retry or contact "
                "is ever permitted, regardless of amount or history. No expected-value "
                "calculation was performed."
            ),
        )

    if retry_count >= MAX_RETRY_COUNT:
        return Decision(
            txn_id=txn_id,
            action="give_up_unlikely",
            predicted_prob=0.0,
            expected_value=0.0,
            reasoning=(
                f"Hard guardrail: retry_count={retry_count} has reached the maximum of "
                f"{MAX_RETRY_COUNT} attempts. Giving up regardless of predicted odds. No "
                "expected-value calculation was performed."
            ),
        )

    # --- Build candidate menu (soft guardrails applied here) ---
    candidates = _candidate_actions(hours_since_failure, retry_count, amount)
    scored = [
        (action, *_score_candidate(action, models, failure_code, hours_since_failure, retry_count, amount))
        for action in candidates
    ]
    scored.sort(key=lambda t: t[2], reverse=True)  # sort by expected_value desc
    best_action, best_prob, best_ev = scored[0]

    # --- Soft guardrail: don't act on a hopeless probability ---
    if best_prob < MIN_PROB_TO_ACT:
        alt_summary = ", ".join(f"{a}(P={p:.2f})" for a, p, _ in scored)
        return Decision(
            txn_id=txn_id,
            action="give_up_unlikely",
            predicted_prob=best_prob,
            expected_value=best_ev,
            reasoning=(
                f"Soft guardrail: even the best candidate '{best_action}' only has "
                f"P(success)={best_prob:.2%}, below the {MIN_PROB_TO_ACT:.0%} action threshold. "
                f"All candidates considered: {alt_summary}. Not worth the contact -> giving up."
            ),
        )

    others = ", ".join(
        f"{a}(P={p:.2f}, EV=Rs.{ev:.2f})" for a, p, ev in scored[1:]
    )
    reasoning = (
        f"Chose '{best_action}' with P(success)={best_prob:.2%} and expected value "
        f"Rs.{best_ev:.2f} on amount Rs.{amount:.2f}, the highest EV among candidates."
    )
    if others:
        reasoning += f" Runner-up alternatives: {others}."

    return Decision(
        txn_id=txn_id,
        action=best_action,
        predicted_prob=best_prob,
        expected_value=best_ev,
        reasoning=reasoning,
    )
