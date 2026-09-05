"""
FastAPI backend for the Revenue Recovery Agent web UI.

Wraps the existing, tested pipeline modules directly - no decision logic is
duplicated here. This file is purely: load data, call the real functions,
stream/serve the results.
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

REVENUE_RECOVERY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REVENUE_RECOVERY_DIR))

from generate_data import (  # noqa: E402
    generate_historical_outcomes,
    generate_failed_payments,
)
from risk_model import fit_risk_models  # noqa: E402
from policy import decide  # noqa: E402
from messenger import generate_message  # noqa: E402
from main import simulate_actual_outcome  # noqa: E402
from calibration_check import run_kfold_cv, summarize_kfold, N_FOLDS, SPLIT_SEED  # noqa: E402

HIST_CSV = REVENUE_RECOVERY_DIR / "historical_outcomes.csv"
BATCH_CSV = REVENUE_RECOVERY_DIR / "failed_payments.csv"
AUDIT_CSV = REVENUE_RECOVERY_DIR / "recovery_audit_trail.csv"

app = FastAPI(title="Revenue Recovery Agent")

# Simple process-wide cache: refit only when the historical CSV's mtime changes.
_model_cache = {"mtime": None, "models": None}
_simulation_rng = np.random.default_rng(2026)


def get_models():
    mtime = HIST_CSV.stat().st_mtime
    if _model_cache["mtime"] != mtime:
        historical = pd.read_csv(HIST_CSV)
        _model_cache["models"] = fit_risk_models(historical)
        _model_cache["mtime"] = mtime
    return _model_cache["models"]


@app.post("/api/generate")
def api_generate():
    """Generate a fresh random historical dataset + current batch.

    Uses an unseeded RNG (unlike generate_data.py's own __main__, which is
    seeded for reproducibility) so each click produces a genuinely different
    batch for the live demo.
    """
    fresh_rng = np.random.default_rng()
    historical = generate_historical_outcomes(rng=fresh_rng)
    batch = generate_failed_payments(rng=fresh_rng)
    historical.to_csv(HIST_CSV, index=False)
    batch.to_csv(BATCH_CSV, index=False)
    # Force a refit on the next /api/run.
    _model_cache["mtime"] = None
    return {
        "historical_rows": len(historical),
        "batch_rows": len(batch),
        "total_at_risk": float(batch["amount"].sum()),
    }


async def _run_stream():
    models = get_models()
    batch = pd.read_csv(BATCH_CSV)

    rows = []
    for _, txn in batch.iterrows():
        txn_dict = txn.to_dict()
        d = decide(txn_dict, models)
        message = generate_message(d.action, txn["preferred_lang"], txn["amount"], txn["failure_code"])
        recovered = simulate_actual_outcome(d.action, txn["failure_code"], int(txn["retry_count"]),
                                             _simulation_rng, txn["payment_method"], bool(txn["is_subscription"]))

        row = {
            "txn_id": d.txn_id,
            "customer_id": txn["customer_id"],
            "amount": float(txn["amount"]),
            "failure_code": txn["failure_code"],
            "action": d.action,
            "predicted_prob": round(d.predicted_prob, 4),
            "expected_value": round(d.expected_value, 2),
            "reasoning": d.reasoning,
            "recovered": bool(recovered),
            "revenue_recovered": float(txn["amount"]) if recovered else 0.0,
            "message": message,
        }
        rows.append(row)

        yield f"event: decision\ndata: {json.dumps(row)}\n\n"
        await asyncio.sleep(0.06)

    audit = pd.DataFrame(rows)
    audit.to_csv(AUDIT_CSV, index=False)

    total_at_risk = float(audit["amount"].sum())
    total_recovered = float(audit["revenue_recovered"].sum())
    action_counts = audit["action"].value_counts().to_dict()
    by_code = audit.groupby("failure_code")["revenue_recovered"].sum().to_dict()

    summary = {
        "transactions": len(audit),
        "total_at_risk": total_at_risk,
        "total_recovered": total_recovered,
        "recovery_rate": (total_recovered / total_at_risk) if total_at_risk else 0.0,
        "action_counts": action_counts,
        "hard_stops": int(action_counts.get("compliance_stop", 0)),
        "soft_giveups": int(action_counts.get("give_up_unlikely", 0)),
        "recovered_by_code": by_code,
    }
    yield f"event: summary\ndata: {json.dumps(summary)}\n\n"


@app.get("/api/run")
async def api_run():
    return StreamingResponse(_run_stream(), media_type="text/event-stream")


@app.get("/api/audit.csv")
def api_audit_csv():
    return FileResponse(AUDIT_CSV, media_type="text/csv", filename="recovery_audit_trail.csv")


@app.get("/api/calibration")
def api_calibration():
    """Same 5-fold CV calibration_check.py's own CLI report uses (via the
    shared run_kfold_cv/summarize_kfold) - this used to run a separate,
    quicker single train/test split, which meant the web UI could show a
    different, less rigorous number than what `python calibration_check.py`
    printed in the terminal. Now both surfaces report identical numbers."""
    historical = pd.read_csv(HIST_CSV)
    fold_results = run_kfold_cv(historical, N_FOLDS, SPLIT_SEED)
    return summarize_kfold(fold_results)


@app.get("/api/guardrail-tests")
def api_guardrail_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_guardrails.py", "-v", "--tb=no"],
        cwd=REVENUE_RECOVERY_DIR,
        capture_output=True,
        text=True,
    )
    # Only the per-test progress lines (e.g. "test_x.py::test_y PASSED [ 14%]") -
    # pytest also repeats each failed test's name in a "short test summary info"
    # footer at the end (just "FAILED test_x.py::test_y", no "%]"), which would
    # otherwise get double-counted as a second test.
    lines = [l for l in result.stdout.splitlines() if ("PASSED" in l or "FAILED" in l) and "%]" in l]
    tests = []
    for line in lines:
        name = line.split("::", 1)[1].split(" ")[0] if "::" in line else line
        tests.append({"name": name, "passed": "PASSED" in line})

    return {
        "all_passed": result.returncode == 0,
        "passed_count": sum(t["passed"] for t in tests),
        "total_count": len(tests),
        "tests": tests,
    }


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
