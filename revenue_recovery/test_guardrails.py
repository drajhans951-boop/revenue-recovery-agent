"""
Guardrail-breaking tests. Each test deliberately builds the MOST favorable
possible circumstances for the guardrail to fail, and asserts it holds anyway.
Fits the risk model on the real generated historical data so these are true
end-to-end guardrail checks, not guardrails tested against a mocked model.
"""
import pandas as pd
import pytest

from risk_model import fit_risk_models
from policy import decide


@pytest.fixture(scope="module")
def models():
    historical = pd.read_csv("historical_outcomes.csv")
    return fit_risk_models(historical)


def test_lost_stolen_card_never_retried_even_with_huge_amount_and_favorable_history(models):
    txn = dict(
        txn_id="BREAK1",
        amount=10_000_000,       # absurdly large amount, would dominate any EV calc
        failure_code="lost_stolen_card",
        hours_since_failure=48,  # "favorable" timing for a normal code
        retry_count=0,           # first attempt, nothing counting against it
    )
    d = decide(txn, models)
    assert d.action == "compliance_stop"
    assert d.predicted_prob == 0.0
    assert d.expected_value == 0.0


def test_mandate_revoked_never_retried_even_with_huge_amount_and_favorable_history(models):
    txn = dict(
        txn_id="BREAK2",
        amount=10_000_000,
        failure_code="mandate_revoked",
        hours_since_failure=48,
        retry_count=0,
    )
    d = decide(txn, models)
    assert d.action == "compliance_stop"
    assert d.predicted_prob == 0.0
    assert d.expected_value == 0.0


@pytest.mark.parametrize("failure_code,amount", [
    ("insufficient_funds", 50000),
    ("network_timeout", 1),
    ("do_not_honor", 999999),
])
def test_max_retries_reached_gives_up_regardless_of_amount_or_code(models, failure_code, amount):
    txn = dict(
        txn_id="BREAK3",
        amount=amount,
        failure_code=failure_code,
        hours_since_failure=0,
        retry_count=3,  # at the max
    )
    d = decide(txn, models)
    assert d.action == "give_up_unlikely"


def test_low_odds_across_the_board_gives_up_instead_of_a_low_odds_retry(models):
    # do_not_honor at hours=24/retry=1 pushes every candidate's probability
    # (even the best one) below the 5% action threshold.
    txn = dict(
        txn_id="BREAK4",
        amount=50000,  # large amount, tempting for EV if the guardrail didn't exist
        failure_code="do_not_honor",
        hours_since_failure=24,
        retry_count=1,
    )
    d = decide(txn, models)
    assert d.action == "give_up_unlikely"
    assert d.predicted_prob < 0.05


def test_cooldown_blocks_immediate_retry_after_a_recent_attempt(models):
    txn = dict(
        txn_id="BREAK5",
        amount=5000,
        failure_code="insufficient_funds",
        hours_since_failure=1,   # only 1 hour since the last attempt
        retry_count=1,           # a prior attempt has already happened
    )
    d = decide(txn, models)
    assert d.action != "retry_0h"
