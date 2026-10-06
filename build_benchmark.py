"""Steps 2-3: extract (drug, condition, dose) triples from raw labels and
generate labeled claims (TRUE + perturbed FALSE), with anti-leakage.

The ground truth of every claim is known by construction. No machine learning:
standard library and regular expressions only. Output: data/benchmark.jsonl.

Slots per triple:
  drug      = openfda.generic_name (+ brand surface) + rxcui
  condition = a phrase mined from indications_and_usage
  dose      = {amount, unit, route, freq_per_day, max_dose, max_unit}

Perturbations (each TRUE spawns up to 3 negatives):
  wrong-drug        -> swap to a drug whose condition set is disjoint
  unsafe-dose       -> amount = max_dose * k  (requires explicit max)
  wrong-indication  -> pair drug with a never-indicated condition
Anti-leakage: a negative is dropped if the (drug, condition) pair is true for
any label in the corpus.
"""
import json
import re
import sys
from collections import defaultdict

RAW = "data/labels_raw.json"
OUT = "data/benchmark.jsonl"

# --- dose parsing ----------------------------------------------------------
UNIT = r"(mg|mcg|g|mL|units?|IU)"
DOSE_RE = re.compile(rf"(\d+(?:\.\d+)?)\s*{UNIT}\b", re.I)
MAX_RE = re.compile(rf"(?:maximum|not to exceed|up to)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*{UNIT}", re.I)
FREQ = [
    (r"\bonce (?:a|per) day|once daily|\bq\.?d\b|\bdaily\b", 1.0),
    (r"\btwice (?:a|per) day|twice daily|\bb\.?i\.?d\b", 2.0),
    (r"\bthree times (?:a|per) day|\bt\.?i\.?d\b", 3.0),
    (r"\bfour times (?:a|per) day|\bq\.?i\.?d\b", 4.0),
    (r"\bevery 12 hours|\bq12h\b", 2.0),
    (r"\bevery 8 hours|\bq8h\b", 3.0),
    (r"\bonce weekly|\bweekly\b", 1 / 7.0),
]
ROUTES = ["oral", "intravenous", "subcutaneous", "topical", "inhaled", "intramuscular"]


def parse_dose(text):
    t = text.replace("\n", " ")
    m = DOSE_RE.search(t)
    if not m:
        return None
    amount, unit = float(m.group(1)), m.group(2).lower()
    freq = next((f for pat, f in FREQ if re.search(pat, t, re.I)), None)
    route = next((r for r in ROUTES if r in t.lower()), None)
    mx = MAX_RE.search(t)
    # The maximum carries its own unit: a label routinely states the two in
    # different ones ("500 mg ... not to exceed 12 g"), so max_dose is only
    # meaningful together with max_unit.
    max_dose = float(mx.group(1)) if mx else None
    max_unit = mx.group(2).lower() if mx else None
    return {
        "amount": amount, "unit": unit, "route": route,
        "freq_per_day": freq, "max_dose": max_dose, "max_unit": max_unit,
        "max_source": "explicit" if max_dose else "none",
    }


def unsafe_dose(dose, factor=5):
    """The unsafe-dose perturbation: exceed the label's stated maximum.

    The inflated amount is expressed in the maximum's own unit, since a label
    often states the dose and the maximum in different units ("500 mg ... not to
    exceed 12 g"). Shared by both benchmark builders.
    """
    return dict(dose, amount=dose["max_dose"] * factor,
                unit=dose.get("max_unit") or dose["unit"])


# --- condition extraction (regex heuristic; build_benchmark_ner.py uses NER) --
STOP = re.compile(r"indications? and usage|is indicated|are indicated|indicated (?:for|as|in)"
                  r"|treatment of|relief of|management of|^\d+\s*", re.I)


def extract_condition(ind_text):
    """Pull a short condition phrase from the indications section with a
    deterministic regular-expression heuristic (build_benchmark_ner.py uses a
    biomedical NER model instead)."""
    t = ind_text.replace("\n", " ")
    # take text after the first "indicated for/as/in ... "
    m = re.search(r"indicated\s+(?:for|as|in)\s+(?:the\s+)?(?:treatment of|relief of|"
                  r"management of|adjunct[^,.]*?(?:control of|with)\s+)?([a-z][a-z0-9 \-/]{4,60})",
                  t, re.I)
    if not m:
        return None
    phrase = m.group(1).strip(" .,-")
    phrase = re.split(r"\b(?:and|or|in adults|in patients|caused by)\b", phrase, 1)[0].strip()
    if len(phrase) < 4 or STOP.search(phrase):
        return None
    return phrase.lower()


def load_triples():
    labels = json.load(open(RAW))
    triples = []
    for r in labels:
        of = r.get("openfda", {})
        generic = (of.get("generic_name") or [None])[0]
        brand = (of.get("brand_name") or [None])[0]
        rxcui = (of.get("rxcui") or [None])[0]
        ind = (r.get("indications_and_usage") or [""])[0]
        dos = (r.get("dosage_and_administration") or [""])[0]
        if not (generic and ind):
            continue
        cond = extract_condition(ind)
        dose = parse_dose(dos)
        if not cond:
            continue
        triples.append({
            "drug": generic.lower(), "brand": (brand or generic).lower(),
            "rxcui": rxcui, "condition": cond, "dose": dose,
            "set_id": of.get("spl_set_id"),
        })
    return triples


def make_claim_text(drug_surface, dose, condition):
    parts = [drug_surface.title()]
    if dose:
        if dose.get("amount"):
            parts.append(f"{dose['amount']:g} {dose['unit']}")
        if dose.get("route"):
            parts.append(dose["route"])
        f = dose.get("freq_per_day")
        if f == 1.0:
            parts.append("once daily")
        elif f == 2.0:
            parts.append("twice daily")
        elif f and f == 3.0:
            parts.append("three times daily")
    parts.append(f"for {condition}")
    return " ".join(parts).strip() + "."


def build():
    triples = load_triples()
    # index: which conditions is each drug truly indicated for (anti-leakage)
    drug_conditions = defaultdict(set)
    for t in triples:
        drug_conditions[t["drug"]].add(t["condition"])
    all_conditions = sorted({t["condition"] for t in triples})
    all_drugs = sorted(drug_conditions.keys())

    claims, cid = [], 0
    dropped_leak = 0

    def emit(text, label, pert, tier, t, drug, cond, dose):
        nonlocal cid
        cid += 1
        claims.append({
            "id": f"clm_{cid:05d}", "claim_text": text, "label": label,
            "perturbation": pert, "tier": tier,
            "drug": {"surface": drug, "rxcui": t["rxcui"]},
            "condition": {"surface": cond},
            "dose": dose,
            "gold_citation": {"set_id": t["set_id"], "section": "indications_and_usage"},
            "source": "openFDA-SPL",
        })

    for i, t in enumerate(triples):
        drug, cond, dose = t["drug"], t["condition"], t["dose"]
        # TRUE
        emit(make_claim_text(t["brand"], dose, cond), "support", None, "E",
             t, drug, cond, dose)

        # wrong-drug: a drug whose true conditions don't include this condition
        for od in all_drugs[(i * 7) % len(all_drugs):]:
            if od != drug and cond not in drug_conditions[od]:
                emit(make_claim_text(od, dose, cond), "refute", "wrong-drug", "E",
                     t, od, cond, dose)
                break

        # unsafe-dose: only when explicit max known
        if dose and dose.get("max_dose") and dose.get("amount"):
            bad = unsafe_dose(dose)
            emit(make_claim_text(t["brand"], bad, cond), "refute", "unsafe-dose", "M",
                 t, drug, cond, bad)

        # wrong-indication: a condition this drug is never indicated for
        for oc in all_conditions[(i * 13) % len(all_conditions):]:
            if oc not in drug_conditions[drug]:
                # anti-leakage: ensure no label asserts (drug, oc)
                emit(make_claim_text(t["brand"], dose, oc), "refute", "wrong-indication",
                     "E", t, drug, oc, dose)
                break
            else:
                dropped_leak += 1

    with open(OUT, "w") as f:
        for c in claims:
            f.write(json.dumps(c) + "\n")

    # summary
    from collections import Counter
    by_label = Counter(c["label"] for c in claims)
    by_pert = Counter(c["perturbation"] for c in claims)
    n_dose = sum(1 for t in triples if t["dose"])
    n_max = sum(1 for t in triples if t["dose"] and t["dose"].get("max_dose"))
    print(f"labels loaded         : {len(json.load(open(RAW)))}")
    print(f"clean triples         : {len(triples)}  (with dose: {n_dose}, explicit max: {n_max})")
    print(f"claims generated      : {len(claims)}")
    print(f"  by label            : {dict(by_label)}")
    print(f"  by perturbation     : {dict(by_pert)}")
    print(f"anti-leakage skips    : {dropped_leak}")
    print(f"-> {OUT}")
    print("\n--- sample claims ---")
    for c in claims[:8]:
        print(f"  [{c['label']:>7}/{str(c['perturbation']):>16}] {c['claim_text']}")


if __name__ == "__main__":
    build()
