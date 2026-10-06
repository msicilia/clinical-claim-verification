"""Step 1: download prescription drug labels (with an RxCUI) from openFDA.

No API key is needed at this volume. Saves the raw labels to data/labels_raw.json.

Run: python fetch_labels.py [N]      (N labels, default 300; the paper uses 1000)
"""
import json
import sys
import time
import urllib.parse
import requests

API = "https://api.fda.gov/drug/label.json"
# prescription drugs that carry an RxCUI (so the OMOP/RxNorm bridge is a join)
QUERY = 'openfda.product_type:"HUMAN PRESCRIPTION DRUG" AND _exists_:openfda.rxcui'
OUT = "data/labels_raw.json"


def fetch(n_total=300, page=100):
    results, skip = [], 0
    while len(results) < n_total:
        params = {
            "search": QUERY,
            "limit": min(page, n_total - len(results)),
            "skip": skip,
        }
        url = f"{API}?{urllib.parse.urlencode(params)}"
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  stop: HTTP {r.status_code} at skip={skip}", file=sys.stderr)
            break
        batch = r.json().get("results", [])
        if not batch:
            break
        results.extend(batch)
        skip += len(batch)
        print(f"  fetched {len(results)}/{n_total}", file=sys.stderr)
        time.sleep(0.3)
    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    labels = fetch(n)
    json.dump(labels, open(OUT, "w"))
    print(f"saved {len(labels)} labels -> {OUT}")
