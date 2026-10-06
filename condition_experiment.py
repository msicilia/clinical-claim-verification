"""C1 corpus and baselines: retrieval where lay and clinical wording diverge.

Unlike FDA labels, which also state their brand names, clinical literature uses
canonical terminology: abstracts say "myocardial infarction", not "heart attack",
whereas patients and generated text often use the lay term.

Corpus    : PubMed abstracts fetched with the canonical term, each tagged with its
            condition (30 conditions; cached in data/abstracts.json).
Queries   : the lay synonym of each condition.
Retriever : TF-IDF cosine (shared by all C1 scripts).
Arms      : raw lay query | naive string cleanup | oracle (the canonical term).
Metric    : P@5, R@10 and MRR for the condition's own abstracts, averaged over
            conditions.

The other C1 scripts import the corpus, the retriever and the evaluation from here.

Run: python condition_experiment.py
"""
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

CACHE = "data/abstracts.json"
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOK = re.compile(r"[a-z0-9]+")

# lay term (the query)            ->  canonical clinical term (used to fetch corpus)
SYNONYMS = {
    "heart attack": "myocardial infarction",
    "high blood pressure": "hypertension",
    "high cholesterol": "hypercholesterolemia",
    "high blood sugar": "diabetes mellitus",
    "low blood sugar": "hypoglycemia",
    "blood clot in the lung": "pulmonary embolism",
    "irregular heartbeat": "atrial fibrillation",
    "kidney failure": "renal insufficiency",
    "liver inflammation": "hepatitis",
    "high potassium": "hyperkalemia",
    "low platelet count": "thrombocytopenia",
    "nosebleed": "epistaxis",
    "whooping cough": "pertussis",
    "german measles": "rubella",
    "hair loss": "alopecia",
    "itchy skin": "pruritus",
    "stroke": "cerebral infarction",
    "blood poisoning": "sepsis",
    "high blood calcium": "hypercalcemia",
    "low blood sodium": "hyponatremia",
    "shortness of breath": "dyspnea",
    "difficulty swallowing": "dysphagia",
    "joint inflammation": "arthritis",
    "low red blood cell count": "anemia",
    "high white blood cell count": "leukocytosis",
    "swelling": "edema",
    "fast heart rate": "tachycardia",
    "slow heart rate": "bradycardia",
    "yellow skin": "jaundice",
    "painful urination": "dysuria",
}

# Naive string normalization: knowledge-free cleanup (lowercase, de-pluralize, light
# suffix stripping). It has no knowledge of lay/clinical synonyms, so it cannot map
# "heart attack" to "myocardial infarction"; it separates the effect of a structured
# vocabulary from that of generic string cleanup.
def naive_normalize(text):
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        for suf in ("ies", "es", "s", "ing", "ed"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return " ".join(out)


def toks(s):
    return TOK.findall(s.lower())


def fetch_corpus(per_condition=12):
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    docs = []
    for lay, canon in SYNONYMS.items():
        params = {"db": "pubmed", "term": f'"{canon}"[Title]', "retmax": per_condition,
                  "retmode": "json", "sort": "relevance"}
        ids = json.load(urllib.request.urlopen(
            f"{ESEARCH}?{urllib.parse.urlencode(params)}", timeout=30)
        )["esearchresult"]["idlist"]
        time.sleep(0.4)
        if not ids:
            continue
        fp = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"}
        xml = urllib.request.urlopen(
            f"{EFETCH}?{urllib.parse.urlencode(fp)}", timeout=60).read().decode("utf-8", "ignore")
        time.sleep(0.4)
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue
        for art in root.iter("PubmedArticle"):
            title = " ".join("".join(t.itertext()) for t in art.iter("ArticleTitle"))
            abst = " ".join("".join(a.itertext()) for a in art.iter("AbstractText"))
            text = f"{title} {abst}".strip()
            if len(text) > 80:
                docs.append({"condition": canon, "lay": lay, "text": text})
        print(f"  {canon:28} abstracts so far: {len(docs)}")
    json.dump(docs, open(CACHE, "w"))
    return docs


def tfidf(docs):
    df = Counter()
    tfs = []
    for d in docs:
        tf = Counter(toks(d["text"]))
        tfs.append(tf)
        for t in tf:
            df[t] += 1
    N = len(docs)
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    def vec(tf):
        v = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in tf.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / nrm for t, x in v.items()}
    return idf, [vec(tf) for tf in tfs]


def qvec(text, idf):
    tf = Counter(toks(text))
    v = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in tf.items()}
    nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / nrm for t, x in v.items()}


def cosine(q, d):
    if len(q) > len(d):
        q, d = d, q
    return sum(w * d.get(t, 0.0) for t, w in q.items())


def evaluate(query, idf, doc_vecs, docs, target_condition):
    qv = qvec(query, idf)
    ranked = sorted(range(len(docs)), key=lambda i: -cosine(qv, doc_vecs[i]))
    rel = [i for i in ranked if docs[i]["condition"] == target_condition]
    total_rel = sum(d["condition"] == target_condition for d in docs)
    p5 = sum(docs[i]["condition"] == target_condition for i in ranked[:5]) / 5
    r10 = sum(docs[i]["condition"] == target_condition for i in ranked[:10]) / max(total_rel, 1)
    first = next((r for r, i in enumerate(ranked, 1)
                  if docs[i]["condition"] == target_condition), len(docs) + 1)
    return p5, r10, 1.0 / first


def main():
    docs = fetch_corpus()
    print(f"\ncorpus: {len(docs)} real PubMed abstracts across "
          f"{len(set(d['condition'] for d in docs))} conditions\n")
    idf, doc_vecs = tfidf(docs)

    conds = sorted(set(d["condition"] for d in docs))
    lay_of = {v: k for k, v in SYNONYMS.items()}
    # three arms: raw lay term | naive string cleanup | oracle (canonical term)
    agg = {"A raw": [0.0, 0.0, 0.0], "B naive-string": [0.0, 0.0, 0.0],
           "C oracle": [0.0, 0.0, 0.0]}
    rows = []
    for canon in conds:
        lay = lay_of[canon]
        a = evaluate(lay, idf, doc_vecs, docs, canon)                    # raw lay
        b = evaluate(naive_normalize(lay), idf, doc_vecs, docs, canon)   # naive cleanup
        c = evaluate(canon, idf, doc_vecs, docs, canon)                  # oracle
        rows.append((lay, canon, a, b, c))
        for arm, v in zip(agg, (a, b, c)):
            agg[arm][0] += v[0]; agg[arm][1] += v[1]; agg[arm][2] += v[2]
    n = len(conds)

    print(f"{'lay query -> canonical':46} {'rawP5':>6} {'naivP5':>6} {'orclP5':>6}")
    print("-" * 70)
    for lay, canon, a, b, c in rows:
        print(f"{(lay+' -> '+canon):46} {a[0]:>6.2f} {b[0]:>6.2f} {c[0]:>6.2f}")
    print("-" * 70)
    print(f"{'arm':22} {'P@5':>7} {'R@10':>7} {'MRR':>7}")
    for arm, (P, R, M) in agg.items():
        print(f"{arm:22} {P/n:>7.3f} {R/n:>7.3f} {M/n:>7.3f}")
    base = agg["A raw"]
    print(f"{'LIFT naive - raw':22} {(agg['B naive-string'][0]-base[0])/n:>+7.3f} "
          f"{(agg['B naive-string'][1]-base[1])/n:>+7.3f} {(agg['B naive-string'][2]-base[2])/n:>+7.3f}")
    print(f"{'LIFT oracle - raw':22} {(agg['C oracle'][0]-base[0])/n:>+7.3f} "
          f"{(agg['C oracle'][1]-base[1])/n:>+7.3f} {(agg['C oracle'][2]-base[2])/n:>+7.3f}")


if __name__ == "__main__":
    main()
