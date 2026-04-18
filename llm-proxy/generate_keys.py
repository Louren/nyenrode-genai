#!/usr/bin/env python3
"""
Generate LiteLLM virtual keys for course students.

Usage:
    python generate_keys.py --students students.txt --proxy http://localhost:4000
    python generate_keys.py --count 120 --proxy http://localhost:4000

Each key gets a $5 budget per 30 days, limited to gpt-4o-mini and gpt-4o.
Output is saved to keys_output.csv.
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
import os


PROXY_URL = "http://localhost:4000"
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "")

# Per-student limits
BUDGET_USD = 5.0        # max spend per period
BUDGET_DURATION = "30d"
RPM_LIMIT = 20          # requests per minute
ALLOWED_MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4", "gpt-4.1-mini"]


def generate_key(alias: str, proxy_url: str, master_key: str) -> dict:
    payload = json.dumps({
        "key_alias": alias,
        "max_budget": BUDGET_USD,
        "budget_duration": BUDGET_DURATION,
        "rpm_limit": RPM_LIMIT,
        "models": ALLOWED_MODELS,
        "metadata": {"course": "agentic-ai", "student": alias},
    }).encode()

    req = urllib.request.Request(
        f"{proxy_url}/key/generate",
        data=payload,
        headers={
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Generate LiteLLM student keys")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--students", help="File with one student name/email per line")
    group.add_argument("--count", type=int, help="Generate N anonymous keys (student-001, ...)")
    parser.add_argument("--proxy", default=PROXY_URL, help="LiteLLM proxy base URL")
    parser.add_argument("--master-key", default=MASTER_KEY, help="LiteLLM master key")
    parser.add_argument("--output", default="keys_output.csv", help="Output CSV file")
    args = parser.parse_args()

    if not args.master_key:
        sys.exit("Error: set LITELLM_MASTER_KEY env var or pass --master-key")

    if args.students:
        aliases = [line.strip() for line in Path(args.students).read_text().splitlines() if line.strip()]
    else:
        aliases = [f"student-{i:03d}" for i in range(1, args.count + 1)]

    print(f"Generating {len(aliases)} keys against {args.proxy} ...")

    rows = []
    for alias in aliases:
        try:
            result = generate_key(alias, args.proxy, args.master_key)
            key = result.get("key", "ERROR")
            rows.append({"alias": alias, "key": key, "budget_usd": BUDGET_USD, "rpm_limit": RPM_LIMIT})
            print(f"  OK  {alias}: {key[:20]}...")
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"  ERR {alias}: HTTP {e.code} — {body[:120]}", file=sys.stderr)
            rows.append({"alias": alias, "key": f"ERROR: {e.code}", "budget_usd": "", "rpm_limit": ""})
        except Exception as e:
            print(f"  ERR {alias}: {e}", file=sys.stderr)
            rows.append({"alias": alias, "key": f"ERROR: {e}", "budget_usd": "", "rpm_limit": ""})
        time.sleep(0.1)  # be polite to the proxy

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alias", "key", "budget_usd", "rpm_limit"])
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if not r["key"].startswith("ERROR"))
    print(f"\nDone: {ok}/{len(aliases)} keys generated → {args.output}")


if __name__ == "__main__":
    main()
