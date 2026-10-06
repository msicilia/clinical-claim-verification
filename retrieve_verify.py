"""Lexical nearest-neighbour baseline (Table 3, first row).

Corpus   : one passage per label = generic name + indications + dosage text.
Retrieve : TF-IDF cosine over the claim text, top 1.
Gate     : SUPPORT iff the retrieved passage mentions the claim's drug and
           (the claim states no dose, or its amount is <= a maximum stated in the
           passage); otherwise FLAG. The retrieved passage is the citation.
Scoring  : against the ground truth assigned when the benchmark was generated.

Run: python retrieve_verify.py [benchmark.jsonl]
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict

RAW = "data/labels_raw.json"
BENCH = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark_ner.jsonl"
TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return TOK.findall(s.lower())


def build_corpus():
    docs = []
    for r in json.load(open(RAW)):
        of = r.get("openfda", {})
        generic = (of.get("generic_name") or [None])[0]
        if not generic:
            continue
        ind = (r.get("indications_and_usage") or [""])[0]
        dos = (r.get("dosage_and_administration") or [""])[0]
        docs.append({
            "set_id": of.get("spl_set_id"),
            "drug": generic.lower(),
            "text": f"{generic}. {ind} {dos}".lower(),
        })
    return docs


def tfidf_index(docs):
    df = Counter()
    doc_tf = []
    for d in docs:
        tf = Counter(toks(d["text"]))
        doc_tf.append(tf)
        for t in tf:
            df[t] += 1
    N = len(docs)
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    def vec(tf):
        v = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}, norm

    doc_vecs = [vec(tf)[0] for tf in doc_tf]
    return idf, doc_vecs


def cosine(qv, dv):
    if len(qv) > len(dv):
        qv, dv = dv, qv
    return sum(w * dv.get(t, 0.0) for t, w in qv.items())


def query_vec(text, idf):
    tf = Counter(toks(text))
    v = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in tf.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}


MAX_RE = re.compile(r"(?:maximum|not to exceed|up to)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(mg|mcg|g)", re.I)


def gate(claim, passage_text):
    """Entailment proxy: drug must be present; dose must not exceed a stated max."""
    drug = claim["drug"]["surface"].split()[0].lower()
    if drug not in passage_text:
        return "refute", "drug-not-in-evidence"
    dose = claim.get("dose")
    if dose and dose.get("amount") and dose.get("unit"):
        for m in MAX_RE.finditer(passage_text):
            mx, unit = float(m.group(1)), m.group(2).lower()
            if unit == dose["unit"].lower() and dose["amount"] > mx:
                return "refute", "dose-exceeds-max"
    return "support", "ok"


def main():
    docs = build_corpus()
    idf, doc_vecs = tfidf_index(docs)
    claims = [json.loads(l) for l in open(BENCH)]

    n = len(claims)
    correct = 0
    cite_attached = cite_correct = 0
    per_pert = defaultdict(lambda: [0, 0])  # [caught/correct, total]

    for c in claims:
        qv = query_vec(c["claim_text"], idf)
        best_i, best_s = max(
            ((i, cosine(qv, dv)) for i, dv in enumerate(doc_vecs)),
            key=lambda x: x[1],
        )
        top = docs[best_i]
        pred, _ = gate(c, top["text"])
        gold = c["label"]
        ok = (pred == gold)
        correct += ok

        key = c["perturbation"] or "TRUE(support)"
        per_pert[key][1] += 1
        per_pert[key][0] += ok

        # citation precision: among supported claims, is the cited label the source label?
        if pred == "support":
            cite_attached += 1
            if top["set_id"] == c["gold_citation"]["set_id"]:
                cite_correct += 1

    pos = [c for c in claims if c["label"] == "support"]
    neg = [c for c in claims if c["label"] == "refute"]
    # recompute catch-rate / etc per class
    print(f"corpus passages       : {len(docs)}")
    print(f"claims evaluated      : {n}")
    print(f"overall label accuracy: {correct/n:.3f}")
    print()
    print("per-type accuracy (pred == gold):")
    for k, (good, tot) in sorted(per_pert.items()):
        print(f"  {k:>20}: {good}/{tot} = {good/tot:.3f}")
    print()
    print(f"hallucination catch-rate (FALSE flagged): "
          f"{sum(1 for c in neg if True)}")
    caught = sum(per_pert[k][0] for k in per_pert if k != 'TRUE(support)')
    tot_neg = sum(per_pert[k][1] for k in per_pert if k != 'TRUE(support)')
    print(f"  = {caught}/{tot_neg} = {caught/tot_neg:.3f}")
    print(f"citation precision (correct cite | attached): "
          f"{cite_correct}/{cite_attached} = {cite_correct/max(cite_attached,1):.3f}")


if __name__ == "__main__":
    main()
