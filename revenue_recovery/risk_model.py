"""
Stage 2 - Diagnosis: P(success | hours_since_failure, retry_count) per failure code.

CRITICAL DESIGN NOTE - why a SEPARATE model per failure code, not one shared
logistic regression with a failure-code dummy variable:

A single shared model has ONE coefficient for "hours_since_failure". But the
time-dynamics are opposite for different codes: insufficient_funds success
probability RISES with hours (people top up their account), while
network_timeout success probability FALLS with hours (it's a transient
glitch, waiting doesn't help). One shared coefficient cannot be positive and
negative at once - a shared model would learn some compromise slope that gets
the direction wrong for at least one code (most likely network_timeout, since
it's outnumbered by the rising codes). Fitting one model per failure code lets
each capture its own, possibly opposite, time-trend correctly. This is a real,
observed failure mode in retry-optimization models, not a hypothetical.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HARD_ZERO_CODES = {"lost_stolen_card", "mandate_revoked"}
MIN_ROWS_FOR_MODEL = 20


class CodeModel:
    """Wraps either a fitted LogisticRegression, a historical-average
    fallback, or a hard-coded zero, behind one predict_proba(hours, retry)."""

    def __init__(self, kind: str, model=None, rate: float = None):
        self.kind = kind  # "logistic" | "average" | "hard_zero"
        self.model = model
        self.rate = rate

    def predict_proba(self, hours_since_failure: float, retry_count: int) -> float:
        if self.kind == "hard_zero":
            return 0.0
        if self.kind == "average":
            return float(self.rate)
        # logistic
        X = np.array([[hours_since_failure, retry_count]])
        return float(self.model.predict_proba(X)[0, 1])


def fit_risk_models(historical_df: pd.DataFrame) -> dict:
    """Fit one CodeModel per failure_code found in the historical data, plus
    hard-coded zeros for the two compliance codes (fit or not fit)."""
    models = {}

    for code in HARD_ZERO_CODES:
        # Compliance rule, not a statistical estimate: never attempt to fit.
        models[code] = CodeModel(kind="hard_zero")

    for code, group in historical_df.groupby("failure_code"):
        if code in HARD_ZERO_CODES:
            continue  # already hard-coded above, ignore any historical rows

        n = len(group)
        outcomes = group["success"].values
        has_variance = len(np.unique(outcomes)) > 1

        if n < MIN_ROWS_FOR_MODEL or not has_variance:
            rate = float(outcomes.mean()) if n > 0 else 0.0
            models[code] = CodeModel(kind="average", rate=rate)
            continue

        X = group[["hours_since_failure", "retry_count"]].values
        y = outcomes
        clf = LogisticRegression()
        clf.fit(X, y)
        models[code] = CodeModel(kind="logistic", model=clf)

    return models


def predict_proba(models: dict, failure_code: str, hours_since_failure: float, retry_count: int) -> float:
    model = models.get(failure_code)
    if model is None:
        return 0.0
    return model.predict_proba(hours_since_failure, retry_count)


def print_sanity_table(models: dict):
    """Print predicted P(success) at hours in {0, 24, 48} for every failure
    code and flag whether the direction matches the intended dynamic."""
    expectations = {
        "insufficient_funds": "UP (rises with time)",
        "network_timeout": "DOWN (decays with time)",
        "bank_declined_generic": "FLAT/slight UP",
        "do_not_honor": "FLAT (very low)",
        "lost_stolen_card": "ZERO always",
        "mandate_revoked": "ZERO always",
    }

    print(f"{'failure_code':<24}{'h=0':>8}{'h=24':>8}{'h=48':>8}   expected direction")
    print("-" * 70)
    all_ok = True
    for code, expected in expectations.items():
        p0 = predict_proba(models, code, 0, 0)
        p24 = predict_proba(models, code, 24, 0)
        p48 = predict_proba(models, code, 48, 0)
        print(f"{code:<24}{p0:>8.3f}{p24:>8.3f}{p48:>8.3f}   {expected}")

        if code in HARD_ZERO_CODES:
            ok = p0 == 0.0 and p24 == 0.0 and p48 == 0.0
        elif "UP" in expected:
            ok = p48 >= p0
        elif "DOWN" in expected:
            ok = p48 <= p0
        else:  # FLAT
            ok = abs(p48 - p0) < 0.15
        all_ok = all_ok and ok

    print("-" * 70)
    if all_ok:
        print("Sanity check PASSED: all directions match intuition.")
    else:
        raise AssertionError("Sanity check FAILED: a model's direction does not match intuition.")


def main():
    historical = pd.read_csv("historical_outcomes.csv")
    models = fit_risk_models(historical)
    print_sanity_table(models)


if __name__ == "__main__":
    main()
