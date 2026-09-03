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

`iter_folds` is the single shared fold-splitting function: risk_model.py's
own feature-comparison diagnostic and the web UI's /api/calibration endpoint
both use it too, so every rigor number in this project is now computed on
the same 5 folds rather than each caller inventing its own split.
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


def iter_folds(df: pd.DataFrame, n_splits: int = N_FOLDS, seed: int = SPLIT_SEED):
    """Yield (train_df, test_df) for each of n_splits folds. The one place
    fold-splitting happens in this project - every cross-validated number
    anywhere (this script's own report, risk_model.py's feature comparison,
    the web UI's /api/calibration) is computed over these same folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(df):
        yield df.iloc[train_idx], df.iloc[test_idx]


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
    fold_results = []

    for train, test in iter_folds(historical_df, n_splits, seed):
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


def summarize_kfold(fold_results: list) -> dict:
    """Aggregate per-fold results into plain, JSON-serializable numbers
    (mean/std across folds). Shared by this script's console report and the
    web UI's /api/calibration endpoint, so both surfaces report the exact
    same numbers instead of the endpoint quietly drifting out of sync."""
    model_accs = np.array([f["model_acc"] for f in fold_results])
    naive_accs = np.array([f["naive_acc"] for f in fold_results])

    all_codes = sorted({code for f in fold_results for code in f["per_code"]})
    per_code = {}
    for code in all_codes:
        m = np.array([f["per_code"][code]["model_acc"] for f in fold_results if code in f["per_code"]])
        n = np.array([f["per_code"][code]["naive_acc"] for f in fold_results if code in f["per_code"]])
        per_code[code] = {
            "folds_present": int(sum(1 for f in fold_results if code in f["per_code"])),
            "model_acc_mean": float(m.mean()),
            "model_acc_std": float(m.std()),
            "naive_acc_mean": float(n.mean()),
            "naive_acc_std": float(n.std()),
            "advantage": float(m.mean() - n.mean()),
        }

    buckets = []
    for label in BUCKET_LABELS:
        ns = np.array([f["bucket_stats"][label]["n"] for f in fold_results])
        preds = np.array([f["bucket_stats"][label]["avg_predicted"] for f in fold_results])
        actuals = np.array([f["bucket_stats"][label]["actual_rate"] for f in fold_results])
        if np.all(ns == 0):
            buckets.append({"bucket": label, "avg_n": 0.0, "avg_predicted": None, "actual_rate": None})
        else:
            buckets.append({
                "bucket": label,
                "avg_n": float(ns.mean()),
                "avg_predicted": float(np.nanmean(preds)),
                "actual_rate": float(np.nanmean(actuals)),
            })

    return {
        "n_folds": len(fold_results),
        "model_acc_mean": float(model_accs.mean()),
        "model_acc_std": float(model_accs.std()),
        "naive_acc_mean": float(naive_accs.mean()),
        "naive_acc_std": float(naive_accs.std()),
        "advantage": float(model_accs.mean() - naive_accs.mean()),
        "per_code": per_code,
        "buckets": buckets,
    }


def print_kfold_report(summary: dict):
    print(f"{summary['n_folds']}-fold cross-validation")
    print("=" * 72)
    print("OVERALL ACCURACY (mean +/- std across folds)")
    print(f"  Model (hours/retry/+features per code): {summary['model_acc_mean']:.3f} +/- {summary['model_acc_std']:.3f}")
    print(f"  Naive baseline (per-code historical rate only): {summary['naive_acc_mean']:.3f} +/- {summary['naive_acc_std']:.3f}")
    print(f"  Model advantage over naive baseline: {summary['advantage']:+.3f}")

    print("\nPER-CODE ACCURACY (mean +/- std across folds where the code appeared)")
    print(f"{'failure_code':<26}{'model acc':>18}{'naive acc':>18}{'advantage':>12}")
    print("-" * 74)
    for code, s in summary["per_code"].items():
        print(f"{code:<26}{s['model_acc_mean']:>10.3f} +/-{s['model_acc_std']:>5.3f}"
              f"{s['naive_acc_mean']:>10.3f} +/-{s['naive_acc_std']:>5.3f}{s['advantage']:>+12.3f}")

    print("\nCALIBRATION TABLE (predicted vs. actual, averaged across folds)")
    print(f"{'bucket':<12}{'avg n/fold':>12}{'avg predicted':>16}{'avg actual rate':>18}")
    print("-" * 58)
    for b in summary["buckets"]:
        if b["avg_predicted"] is None:
            print(f"{b['bucket']:<12}{0:>12}{'--':>16}{'--':>18}")
        else:
            print(f"{b['bucket']:<12}{b['avg_n']:>12.1f}{b['avg_predicted']:>16.3f}{b['actual_rate']:>18.3f}")


def main():
    historical = pd.read_csv("historical_outcomes.csv")
    print(f"Total historical rows: {len(historical)}")
    fold_results = run_kfold_cv(historical, N_FOLDS, SPLIT_SEED)
    print_kfold_report(summarize_kfold(fold_results))


if __name__ == "__main__":
    main()
