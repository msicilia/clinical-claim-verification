"""Benchmark generator with biomedical NER condition extraction.

Builds claims exactly as build_benchmark.py does, but extracts the condition from
the indications section with a biomedical NER model rather than a regular
expression. This is the benchmark used by all C2 experiments.
Output: data/benchmark_ner.jsonl

Run: python build_benchmark_ner.py
"""
import json
import re
import warnings
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")
from transformers import pipeline

from build_benchmark import parse_dose, make_claim_text, unsafe_dose

RAW = "data/labels_raw.json"
OUT = "data/benchmark_ner.jsonl"
NER_MODEL = "d4data/biomedical-ner-all"
DISEASE_GROUPS = {"Disease_disorder", "Sign_symptom"}


def clean_condition(entities):
    """Pick the best disease mention: filter subword fragments, prefer the longest
    multi-word clinical phrase."""
    cands = []
    for e in entities:
        w = e["word"].strip()
        if e["entity_group"] not in DISEASE_GROUPS:
            continue
        if "##" in w or len(w) < 5 or not re.search(r"[a-z]{4}", w.lower()):
            continue
        cands.append(w.lower())
    if not cands:
        return None
    # prefer the longest (most specific) phrase
    return max(cands, key=lambda s: (len(s.split()), len(s)))


def load_triples():
    labels = json.load(open(RAW))
    print(f"loading NER model {NER_MODEL} ...")
    ner = pipeline("token-classification", model=NER_MODEL, aggregation_strategy="simple")

    # batch the indication texts for speed
    recs = []
    for r in labels:
        of = r.get("openfda", {})
        g = (of.get("generic_name") or [None])[0]
        ind = (r.get("indications_and_usage") or [""])[0]
        if not (g and ind):
            continue
        recs.append((r, of, g, ind))
    inds = [ind[:400] for _, _, _, ind in recs]
    print(f"running NER on {len(inds)} indications ...")
    all_ents = ner(inds, batch_size=16)

    triples = []
    for (r, of, g, ind), ents in zip(recs, all_ents):
        cond = clean_condition(ents)
        if not cond:
            continue
        dos = (r.get("dosage_and_administration") or [""])[0]
        brand = (of.get("brand_name") or [g])[0]
        triples.append({
            "drug": g.lower(), "brand": (brand or g).lower(),
            "rxcui": (of.get("rxcui") or [None])[0],
            "condition": cond, "dose": parse_dose(dos),
            "set_id": of.get("spl_set_id"),
        })
    return triples


def build():
    triples = load_triples()
    drug_conditions = defaultdict(set)
    for t in triples:
        drug_conditions[t["drug"]].add(t["condition"])
    all_conditions = sorted({t["condition"] for t in triples})
    all_drugs = sorted(drug_conditions.keys())

    claims, cid, dropped = [], 0, 0

    def emit(text, label, pert, tier, t, drug, cond, dose):
        nonlocal cid
        cid += 1
        claims.append({
            "id": f"clm_{cid:05d}", "claim_text": text, "label": label,
            "perturbation": pert, "tier": tier,
            "drug": {"surface": drug, "rxcui": t["rxcui"]},
            "condition": {"surface": cond}, "dose": dose,
            "gold_citation": {"set_id": t["set_id"], "section": "indications_and_usage"},
            "source": "openFDA-SPL+NER",
        })

    for i, t in enumerate(triples):
        drug, cond, dose = t["drug"], t["condition"], t["dose"]
        emit(make_claim_text(t["brand"], dose, cond), "support", None, "E", t, drug, cond, dose)
        for od in all_drugs[(i * 7) % len(all_drugs):]:
            if od != drug and cond not in drug_conditions[od]:
                emit(make_claim_text(od, dose, cond), "refute", "wrong-drug", "E", t, od, cond, dose)
                break
        if dose and dose.get("max_dose") and dose.get("amount"):
            bad = unsafe_dose(dose)
            emit(make_claim_text(t["brand"], bad, cond), "refute", "unsafe-dose", "M", t, drug, cond, bad)
        for oc in all_conditions[(i * 13) % len(all_conditions):]:
            if oc not in drug_conditions[drug]:
                emit(make_claim_text(t["brand"], dose, oc), "refute", "wrong-indication", "E", t, drug, oc, dose)
                break
            else:
                dropped += 1

    with open(OUT, "w") as f:
        for c in claims:
            f.write(json.dumps(c) + "\n")

    by_label = Counter(c["label"] for c in claims)
    by_pert = Counter(c["perturbation"] for c in claims)
    print(f"\nclean triples       : {len(triples)}")
    print(f"claims generated    : {len(claims)}  by_label={dict(by_label)}")
    print(f"  by perturbation   : {dict(by_pert)}")
    print(f"anti-leakage skips  : {dropped}")
    print(f"-> {OUT}\n--- sample conditions (NER) ---")
    seen = set()
    for c in claims:
        if c["label"] == "support" and c["condition"]["surface"] not in seen:
            seen.add(c["condition"]["surface"])
            print("   -", repr(c["condition"]["surface"]))
        if len(seen) >= 16:
            break


if __name__ == "__main__":
    build()
