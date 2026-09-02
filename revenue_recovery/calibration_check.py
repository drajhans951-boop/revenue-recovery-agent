"""
Accuracy & calibration evaluation for the risk model, via 5-fold cross-validation.

A single 80/20 split can be lucky or unlucky - one particular held-out slice
might happen to be easy or hard. 5-fold CV refits all per-code models fresh on
each fold's training portion (respecting the per-code architecture - each
fold's fit_risk_models() call fits one model per failure_code independently,
exactly like a single-split fit would) and reports the MEAN and STANDARD
DEVIATION of accuracy across folds, not just one number.

A "naive" baseline - predict each failure code's own overall historical
average rate, ignoring hours/retry/features entirely - is evaluated with the
same folds, so it's clear whether the model is adding value over a plain
lookup table or not.

`train_test_split`/`evaluate`/`TEST_FRACTION`/`SPLIT_SEED`/`BUCKET_EDGES`/
`BUCKET_LABELS` are kept as-is below (not removed) because risk_model.py's
own feature-comparison diagnostic and the web UI's /api/calibration endpoint
both still use a single split for a fast, one-shot check - the 5-fold version
here is the more rigorous, slower report this script itself now produces.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from risk_model import fit_risk_models, predict_proba

TEST_FRACTION = 0.2
SPLIT_SEED = 7
N_FOLDS = 5

BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BUCKET_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def train_test_split(df: pd.DataFrame, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = int(len(shuffled) * test_fraction)
    test = shuffled.iloc[:n_test]
    train = shuffled.iloc[n_test:]
    return train, test


def evaluate(models: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    preds = [
        predict_proba(models, row.failure_code, row.hours_since_failure, row.retry_count,
                      row.payment_method, row.is_subscription)
        for row in test_df.itertuples()
    ]
    out = test_df.copy()
    out["predicted_prob"] = preds
    return out


def _naive_baseline_probs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """The naive predictor: each code's own historical average rate from the
    fold's TRAIN portion, ignoring hours/retry/features entirely."""
    rates = train_df.groupby("failure_code")["success"].mean()
    return test_df["failure_code"].map(rates).fillna(0.0).values


def _accuracy(predicted_prob, actual) -> float:
    predicted_class = (np.asarray(predicted_prob) >= 0.5).astype(int)
    return float((predicted_class == np.asarray(actual)).mean())


def run_kfold_cv(historical_df: pd.DataFrame, n_splits: int = N_FOLDS, seed: int = SPLIT_SEED) -> list:
    """Run n_splits-fold CV. Returns one result dict per fold with overall
    and per-code accuracy (model vs. naive baseline) and a calibration
    bucket table, all computed on that fold's held-out portion only."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for train_idx, test_idx in kf.split(historical_df):
        train = historical_df.iloc[train_idx]
        test = historical_df.iloc[test_idx]

        models = fit_risk_models(train)
        evaluated = evaluate(models, test)
        naive_probs = _naive_baseline_probs(train, test)

        model_acc = _accuracy(evaluated["predicted_prob"], evaluated["success"])
        naive_acc = _accuracy(naive_probs, test["success"].values)

        per_code = {}
        for code in sorted(test["failure_code"].unique()):
            mask = (evaluated["failure_code"] == code).values
            per_code[code] = {
                "n": int(mask.sum()),
                "model_acc": _accuracy(evaluated["predicted_prob"].values[mask], evaluated["success"].values[mask]),
                "naive_acc": _accuracy(naive_probs[mask], test["success"].values[mask]),
            }

        buckets = pd.cut(evaluated["predicted_prob"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, include_lowest=True)
        bucket_stats = {}
        for label in BUCKET_LABELS:
            subset = evaluated[buckets == label]
            if len(subset) == 0:
                bucket_stats[label] = {"n": 0, "avg_predicted": np.nan, "actual_rate": np.nan}
            else:
                bucket_stats[label] = {
                    "n": len(subset),
                    "avg_predicted": float(subset["predicted_prob"].mean()),
                    "actual_rate": float(subset["success"].mean()),
                }

        fold_results.append({
            "model_acc": model_acc,
            "naive_acc": naive_acc,
            "per_code": per_code,
            "bucket_stats": bucket_stats,
        })

    return fold_results


def print_kfold_report(fold_results: list):
    n_folds = len(fold_results)

    model_accs = np.array([f["model_acc"] for f in fold_results])
    naive_accs = np.array([f["naive_acc"] for f in fold_results])

    print(f"{N_FOLDS}-fold cross-validation ({n_folds} folds actually run)")
    print("=" * 72)
    print("OVERALL ACCURACY (mean +/- std across folds)")
    print(f"  Model (hours/retry/+features per code): {model_accs.mean():.3f} +/- {model_accs.std():.3f}")
    print(f"  Naive baseline (per-code historical rate only): {naive_accs.mean():.3f} +/- {naive_accs.std():.3f}")
    print(f"  Model advantage over naive baseline: {(model_accs.mean() - naive_accs.mean()):+.3f}")

    all_codes = sorted({code for f in fold_results for code in f["per_code"]})
    print("\nPER-CODE ACCURACY (mean +/- std across folds where the code appeared)")
    print(f"{'failure_code':<26}{'model acc':>18}{'naive acc':>18}{'advantage':>12}")
    print("-" * 74)
    for code in all_codes:
        m = np.array([f["per_code"][code]["model_acc"] for f in fold_results if code in f["per_code"]])
        n = np.array([f["per_code"][code]["naive_acc"] for f in fold_results if code in f["per_code"]])
        print(f"{code:<26}{m.mean():>10.3f} +/-{m.std():>5.3f}{n.mean():>10.3f} +/-{n.std():>5.3f}"
              f"{(m.mean() - n.mean()):>+12.3f}")

    print("\nCALIBRATION TABLE (predicted vs. actual, averaged across folds)")
    print(f"{'bucket':<12}{'avg n/fold':>12}{'avg predicted':>16}{'avg actual rate':>18}")
    print("-" * 58)
    for label in BUCKET_LABELS:
        ns = np.array([f["bucket_stats"][label]["n"] for f in fold_results])
        preds = np.array([f["bucket_stats"][label]["avg_predicted"] for f in fold_results])
        actuals = np.array([f["bucket_stats"][label]["actual_rate"] for f in fold_results])
        if np.all(ns == 0):
            print(f"{label:<12}{0:>12}{'--':>16}{'--':>18}")
            continue
        print(f"{label:<12}{ns.mean():>12.1f}{np.nanmean(preds):>16.3f}{np.nanmean(actuals):>18.3f}")


def main():
    historical = pd.read_csv("historical_outcomes.csv")
    print(f"Total historical rows: {len(historical)}")
    fold_results = run_kfold_cv(historical, N_FOLDS, SPLIT_SEED)
    print_kfold_report(fold_results)


if __name__ == "__main__":
    main()
