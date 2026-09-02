"""
OPTIONAL, standalone script - has nothing to do with the synthetic pipeline
in generate_data.py/risk_model.py/policy.py/etc, and nothing else in this
project imports it.

Pulls a small number of REAL Razorpay Test Mode error response payloads, so
they can be inspected alongside the synthetic data (see the "Data
provenance" section of README.md). Uses:
  - Razorpay's documented UPI Collect "always fails" test trick: requesting
    a payment from the VPA `failure@razorpay` deterministically fails on the
    Test Mode S2S (server-to-server) UPI Collect endpoint.
  - A small test-mode Order, created first (required before a payment can
    reference it).

Requires RAZORPAY_TEST_KEY_ID / RAZORPAY_TEST_KEY_SECRET (a free Test Mode
key pair from the Razorpay Dashboard). If either is missing, this script
prints a clear message and exits immediately - no network call is attempted.
It also has NOT been run end-to-end against the live API (no keys were
available while building this project), so treat it as a best-effort,
untested-against-the-live-API starting point, not a verified integration.

Usage:
    export RAZORPAY_TEST_KEY_ID=rzp_test_xxxxxxxx
    export RAZORPAY_TEST_KEY_SECRET=xxxxxxxxxxxxxxxx
    python fetch_test_mode_sample.py
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

API_BASE = "https://api.razorpay.com/v1"
OUTPUT_PATH = "test_mode_sample.json"


def _auth_header(key_id: str, key_secret: str) -> str:
    token = base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
    return f"Basic {token}"


def _post(path: str, payload: dict, auth_header: str) -> dict:
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": auth_header},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"status": resp.status, "body": json.loads(resp.read().decode())}
    except urllib.error.HTTPError as e:
        # A failure response IS the interesting payload here - Razorpay
        # returns error details as a JSON body even on 4xx/5xx.
        return {"status": e.code, "body": json.loads(e.read().decode())}


def main():
    key_id = os.environ.get("RAZORPAY_TEST_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_TEST_KEY_SECRET")

    if not key_id or not key_secret:
        print("SKIPPED: RAZORPAY_TEST_KEY_ID / RAZORPAY_TEST_KEY_SECRET are not both set.")
        print("No network call was attempted. Set both env vars and re-run to fetch real")
        print("Razorpay Test Mode error samples.")
        return

    auth_header = _auth_header(key_id, key_secret)
    results = {}

    print("Creating a small Test Mode order...")
    order_result = _post("/orders", {"amount": 100, "currency": "INR", "receipt": "provenance_sample"}, auth_header)
    results["order"] = order_result
    print(f"  -> status {order_result['status']}")

    order_id = order_result["body"].get("id")
    if order_id:
        print("Attempting a UPI Collect payment against failure@razorpay (documented always-fails VPA)...")
        payment_result = _post(
            "/payments/create/upi",
            {
                "amount": 100,
                "currency": "INR",
                "email": "provenance-sample@example.com",
                "contact": "9999999999",
                "order_id": order_id,
                "method": "upi",
                "vpa": "failure@razorpay",
            },
            auth_header,
        )
        results["upi_collect_payment"] = payment_result
        print(f"  -> status {payment_result['status']}")
    else:
        print("Skipping the payment attempt - order creation did not return an id.")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw response payload(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
