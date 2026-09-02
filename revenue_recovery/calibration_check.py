"""
Accuracy & calibration evaluation for the risk model.

Splits historical_outcomes.csv 80/20, refits the per-code models (reusing
risk_model.fit_risk_models) on the TRAIN split only, then evaluates on the
held-out TEST split:
  - overall accuracy at a 0.5 threshold
  - a calibration table: does "70% predicted" actually mean "succeeds ~70%
    of the time" among the held-out transactions that fell in that bucket?
"""
import numpy as np
import pandas as pd

from risk_model import fit_risk_models, predict_proba

TEST_FRACTION = 0.2
SPLIT_SEED = 7

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
        predict_proba(models, row.failure_code, row.hours_since_failure, row.retry_count)
        for row in test_df.itertuples()
    ]
    out = test_df.copy()
    out["predicted_prob"] = preds
    return out


def print_accuracy(evaluated: pd.DataFrame):
    predicted_class = (evaluated["predicted_prob"] >= 0.5).astype(int)
    accuracy = (predicted_class == evaluated["success"]).mean()
    print(f"Held-out test set size: {len(evaluated)}")
    print(f"Overall accuracy (threshold=0.5): {accuracy:.3f}")


def print_calibration_table(evaluated: pd.DataFrame):
    print("\nCalibration table (held-out test set):")
    print(f"{'bucket':<12}{'n':>6}{'avg predicted':>16}{'actual success rate':>22}")
    print("-" * 56)

    buckets = pd.cut(evaluated["predicted_prob"], bins=BUCKET_EDGES, labels=BUCKET_LABELS, include_lowest=True)
    for label in BUCKET_LABELS:
        subset = evaluated[buckets == label]
        n = len(subset)
        if n == 0:
            print(f"{label:<12}{n:>6}{'--':>16}{'--':>22}")
            continue
        avg_pred = subset["predicted_prob"].mean()
        actual_rate = subset["success"].mean()
        print(f"{label:<12}{n:>6}{avg_pred:>16.3f}{actual_rate:>22.3f}")


def main():
    historical = pd.read_csv("historical_outcomes.csv")
    train, test = train_test_split(historical, TEST_FRACTION, SPLIT_SEED)

    print(f"Train rows: {len(train)}  Test rows: {len(test)}")
    models = fit_risk_models(train)

    evaluated = evaluate(models, test)
    print_accuracy(evaluated)
    print_calibration_table(evaluated)


if __name__ == "__main__":
    main()
