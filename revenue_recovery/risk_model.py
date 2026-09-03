"""
Stage 2 - Diagnosis: P(success | hours_since_failure, retry_count, payment_method,
is_subscription) per failure code.

CRITICAL DESIGN NOTE - why a SEPARATE model per failure code, not one shared
logistic regression with a failure-code dummy variable:

A single shared model has ONE coefficient for "hours_since_failure". But the
time-dynamics are opposite for different codes: insufficient_funds success
probability RISES with hours (people top up their account), while
payment_timed_out success probability FALLS with hours (it's a transient
glitch, waiting doesn't help). One shared coefficient cannot be positive and
negative at once - a shared model would learn some compromise slope that gets
the direction wrong for at least one code (most likely payment_timed_out,
since it's outnumbered by the rising codes). Fitting one model per failure
code lets each capture its own, possibly opposite, time-trend correctly. This
is a real, observed failure mode in retry-optimization models, not a
hypothetical.

Feature set: `payment_method` and `is_subscription` were added on top of the
original `hours_since_failure`/`retry_count` pair. `payment_method` is
one-hot encoded across a FIXED category list (not derived from whatever
happens to be in a given training slice) so the feature vector always has the
same shape at predict time regardless of which categories a particular
failure code's training rows happened to contain.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HARD_ZERO_CODES = {"debit_instrument_blocked", "mandate_revoked"}
MIN_ROWS_FOR_MODEL = 20
# A code needs at least this many examples of BOTH outcomes before its extra
# payment_method/is_subscription one-hot columns are trusted. Below this, a
# handful of positive examples can make those columns fit spurious
# correlations and destabilize even the hours_since_failure coefficient -
# exactly the kind of noise-fitting we don't want. Below the threshold we
# silently fall back to the base 2 features for that code, even when the
# caller asked for use_extra_features=True.
MIN_MINORITY_CLASS_FOR_EXTRA_FEATURES = 25

# Fixed so train-time and predict-time one-hot vectors always line up, even if
# a particular failure code's training rows never contain every category.
PAYMENT_METHOD_CATEGORIES = ["upi", "card", "netbanking", "wallet"]


def _build_features(hours_since_failure, retry_count, payment_method, is_subscription,
                     use_extra_features: bool) -> list:
    """Build one feature row. Shared by both fit (vectorized via DataFrame.apply
    semantics, called per-row below) and predict (single row) so the two can
    never drift apart."""
    features = [float(hours_since_failure), float(retry_count)]
    if use_extra_features:
        features.append(1.0 if is_subscription else 0.0)
        for category in PAYMENT_METHOD_CATEGORIES:
            features.append(1.0 if payment_method == category else 0.0)
    return features


def _build_feature_matrix(df: pd.DataFrame, use_extra_features: bool) -> np.ndarray:
    rows = [
        _build_features(r.hours_since_failure, r.retry_count, r.payment_method, r.is_subscription,
                         use_extra_features)
        for r in df.itertuples()
    ]
    return np.array(rows)


class CodeModel:
    """Wraps either a fitted LogisticRegression, a historical-average
    fallback, or a hard-coded zero, behind one predict_proba(...)."""

    def __init__(self, kind: str, model=None, rate: float = None, use_extra_features: bool = True):
        self.kind = kind  # "logistic" | "average" | "hard_zero"
        self.model = model
        self.rate = rate
        self.use_extra_features = use_extra_features

    def predict_proba(self, hours_since_failure: float, retry_count: int,
                       payment_method: str = "upi", is_subscription: bool = False) -> float:
        if self.kind == "hard_zero":
            return 0.0
        if self.kind == "average":
            return float(self.rate)
        # logistic - build the feature vector in the same mode this model was fit in.
        X = np.array([_build_features(hours_since_failure, retry_count, payment_method,
                                       is_subscription, self.use_extra_features)])
        return float(self.model.predict_proba(X)[0, 1])


def fit_risk_models(historical_df: pd.DataFrame, use_extra_features: bool = True) -> dict:
    """Fit one CodeModel per failure_code found in the historical data, plus
    hard-coded zeros for the two compliance codes (fit or not fit).

    use_extra_features=True (the production default, used by policy/main/webapp)
    fits on [hours, retry, is_subscription, payment_method one-hots].
    use_extra_features=False fits on just [hours, retry] - the original,
    smaller feature set - and exists only so risk_model.py's own diagnostic
    below can honestly compare "with vs without the new features".
    """
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
            models[code] = CodeModel(kind="average", rate=rate, use_extra_features=use_extra_features)
            continue

        minority_count = min((outcomes == 0).sum(), (outcomes == 1).sum())
        code_use_extra = use_extra_features and minority_count >= MIN_MINORITY_CLASS_FOR_EXTRA_FEATURES

        X = _build_feature_matrix(group, code_use_extra)
        y = outcomes
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        models[code] = CodeModel(kind="logistic", model=clf, use_extra_features=code_use_extra)

    return models


def predict_proba(models: dict, failure_code: str, hours_since_failure: float, retry_count: int,
                   payment_method: str = "upi", is_subscription: bool = False) -> float:
    model = models.get(failure_code)
    if model is None:
        return 0.0
    return model.predict_proba(hours_since_failure, retry_count, payment_method, is_subscription)


def print_sanity_table(models: dict):
    """Print predicted P(success) at hours in {0, 24, 48} for every failure
    code and flag whether the direction matches the intended dynamic."""
    expectations = {
        "insufficient_funds": "UP (rises with time)",
        "payment_timed_out": "DOWN (decays with time)",
        "card_declined": "FLAT/slight UP",
        "do_not_honor": "FLAT (very low)",
        "debit_instrument_blocked": "ZERO always",
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


def _accuracy(models: dict, df: pd.DataFrame) -> float:
    preds = [
        predict_proba(models, r.failure_code, r.hours_since_failure, r.retry_count,
                      r.payment_method, r.is_subscription)
        for r in df.itertuples()
    ]
    predicted_class = (np.array(preds) >= 0.5).astype(int)
    return float((predicted_class == df["success"].values).mean())


def print_feature_comparison(historical: pd.DataFrame):
    """Diagnostic: does adding payment_method/is_subscription actually help,
    per failure code - or are we just fitting noise? Runs the SAME 5-fold CV
    calibration_check.py uses (via its shared iter_folds), refitting both a
    basic and an extended model set per fold, so this is a properly
    cross-validated comparison rather than one split's numbers - a single
    split is exactly the "lucky/unlucky" problem calibration_check.py's own
    k-fold report exists to avoid, so this diagnostic shouldn't have a
    weaker standard of evidence than that one."""
    from calibration_check import iter_folds, N_FOLDS, SPLIT_SEED

    # code -> list of (acc_basic, acc_extended, n, was_gated_off), one entry per fold
    per_code_folds = {}
    overall_basic_accs = []
    overall_extended_accs = []

    for train, test in iter_folds(historical, N_FOLDS, SPLIT_SEED):
        basic_models = fit_risk_models(train, use_extra_features=False)
        extended_models = fit_risk_models(train, use_extra_features=True)

        overall_basic_accs.append(_accuracy(basic_models, test))
        overall_extended_accs.append(_accuracy(extended_models, test))

        for code in sorted(test["failure_code"].unique()):
            subset = test[test["failure_code"] == code]
            acc_basic = _accuracy(basic_models, subset)
            acc_extended = _accuracy(extended_models, subset)
            model = extended_models.get(code)
            was_gated_off = model is not None and model.kind == "logistic" and not model.use_extra_features
            per_code_folds.setdefault(code, []).append((acc_basic, acc_extended, len(subset), was_gated_off))

    print(f"\nFeature comparison ({N_FOLDS}-fold CV): original 2 features vs. +payment_method/+is_subscription")
    print(f"{'failure_code':<26}{'acc_basic':>16}{'acc_extended':>18}{'delta':>10}{'gated':>8}")
    print("-" * 80)

    no_improvement_notes = []
    partially_gated_notes = []
    for code, folds in per_code_folds.items():
        basics = np.array([f[0] for f in folds])
        extendeds = np.array([f[1] for f in folds])
        gated_count = sum(1 for f in folds if f[3])
        delta = extendeds.mean() - basics.mean()

        gated_label = f"{gated_count}/{len(folds)}" if gated_count else "no"
        print(f"{code:<26}{basics.mean():>9.3f} +/-{basics.std():>4.2f}"
              f"{extendeds.mean():>11.3f} +/-{extendeds.std():>4.2f}{delta:>+10.3f}{gated_label:>8}")

        if 0 < gated_count < len(folds):
            partially_gated_notes.append(f"{code} ({gated_count}/{len(folds)} folds)")
        elif gated_count == 0 and delta <= 1e-9:
            no_improvement_notes.append(code)

    overall_basic = np.array(overall_basic_accs)
    overall_extended = np.array(overall_extended_accs)
    print("-" * 80)
    print(f"{'OVERALL':<26}{overall_basic.mean():>9.3f} +/-{overall_basic.std():>4.2f}"
          f"{overall_extended.mean():>11.3f} +/-{overall_extended.std():>4.2f}"
          f"{(overall_extended.mean() - overall_basic.mean()):>+10.3f}")

    fully_gated = [c for c, folds in per_code_folds.items() if all(f[3] for f in folds)]
    if fully_gated:
        print(f"\nNote: extra features were AUTOMATICALLY DISABLED in every fold for: {', '.join(fully_gated)}"
              f" (fewer than {MIN_MINORITY_CLASS_FOR_EXTRA_FEATURES} examples of the minority outcome"
              " class in that fold's training data - not enough support to trust the extra one-hot"
              " columns, so these fall back to the original 2-feature model even in 'extended' mode).")
    if partially_gated_notes:
        print(f"\nNote: extra features were gated off in SOME folds only for: {', '.join(partially_gated_notes)}"
              " - this code sits right at the minority-class support threshold, so whether the richer"
              " model is even attempted can depend on which rows a given fold happened to draw.")
    if no_improvement_notes:
        print(f"\nNote: no accuracy improvement from the new features (features used, still no gain) for: "
              f"{', '.join(no_improvement_notes)}. This is expected for card_declined (the synthetic ground"
              " truth deliberately does not vary by payment_method/is_subscription for this code) and"
              " trivially expected for the hard-zero compliance codes - not force-fitting noise.")


def main():
    historical = pd.read_csv("historical_outcomes.csv")
    models = fit_risk_models(historical)
    print_sanity_table(models)
    print_feature_comparison(historical)


if __name__ == "__main__":
    main()
