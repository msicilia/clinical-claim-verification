"""Brand-to-generic normalization against drug labels (Section 4.1, first result).

A query may use the brand name ("Mekinist") while the evidence is indexed under the
generic name ("trametinib"); the two share no words. OMOP/RxNorm provides the
brand -> generic mapping through the drug's RxCUI; here the lookup is simulated with
a brand -> generic table built from the labels themselves.

Two arms, identical retriever (TF-IDF), identical corpus (generic-named passages):
  Arm A (raw)        : query uses the brand surface form, as written.
  Arm B (normalized) : brand surface replaced by its generic via the lookup.

Metric: where does the CORRECT label rank for each query?
  Recall@1, Recall@3, MRR -- over the messy subset (brand != generic), where
  normalization can possibly matter, and over ALL support claims (to show it does
  not hurt already-clean inputs).
"""
import json
import math
import re
from collections import Counter

RAW = "data/labels_raw.json"
TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return TOK.findall(s.lower())


def load():
    labels = json.load(open(RAW))
    docs, brand2generic, triples = [], {}, []
    for r in labels:
        of = r.get("openfda", {})
        g = (of.get("generic_name") or [None])[0]
        b = (of.get("brand_name") or [None])[0]
        ind = (r.get("indications_and_usage") or [""])[0]
        dos = (r.get("dosage_and_administration") or [""])[0]
        sid = of.get("spl_set_id")
        if not (g and ind and sid):
            continue
        g = g.strip().lower()
        # canonical corpus passage uses the GENERIC name
        docs.append({"set_id": sid, "drug": g,
                     "text": f"{g}. {ind} {dos}".lower()})
        if b and b.strip().lower() != g:
            brand2generic[b.strip().lower()] = g
        # one support "claim" per label: the drug stated under its brand surface
        surface = (b or g).strip().lower()
        triples.append({"set_id": sid, "surface": surface, "generic": g,
                        "messy": surface != g, "ind": ind})
    return docs, brand2generic, triples


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


def normalize(text, brand2generic):
    """The OMOP step: replace any known brand surface with its generic."""
    t = text.lower()
    for brand, generic in brand2generic.items():
        if brand in t:
            t = t.replace(brand, generic)
    return t


def rank_of_gold(query, idf, doc_vecs, docs, gold_sid):
    qv = qvec(query, idf)
    scored = sorted(
        ((cosine(qv, dv), docs[i]["set_id"]) for i, dv in enumerate(doc_vecs)),
        key=lambda x: -x[0],
    )
    for rank, (_, sid) in enumerate(scored, 1):
        if sid == gold_sid:
            return rank
    return len(scored) + 1


def metrics(ranks):
    n = len(ranks)
    if n == 0:
        return {"n": 0, "R@1": 0, "R@3": 0, "MRR": 0}
    return {
        "n": n,
        "R@1": sum(r == 1 for r in ranks) / n,
        "R@3": sum(r <= 3 for r in ranks) / n,
        "MRR": sum(1.0 / r for r in ranks) / n,
    }


def main():
    docs, brand2generic, triples = load()
    idf, doc_vecs = tfidf(docs)
    print(f"corpus passages          : {len(docs)}")
    print(f"brand->generic synonyms  : {len(brand2generic)}")
    print(f"support claims total     : {len(triples)}")
    messy = [t for t in triples if t["messy"]]
    print(f"  of which 'messy'       : {len(messy)} (brand != generic)\n")

    # Two query styles to show the effect depends on how much the ENTITY carries
    # the retrieval signal:
    #   short  = "<drug>"                 -> realistic user query; entity IS the signal
    #   context= "<drug> for <indication>"-> query quotes the label; entity barely matters
    query_styles = {
        "short  (entity-dominated)": lambda t, surface: surface,
        "context (quotes the label)": lambda t, surface: f"{surface} for {t['ind'][:120]}",
    }

    for style_name, make_q in query_styles.items():
        print(f"\n================  QUERY STYLE: {style_name}  ================")
        print(f"{'subset':34} {'arm':18} {'n':>4} {'R@1':>6} {'R@3':>6} {'MRR':>6}")
        print("-" * 80)
        for subset_name, subset in [("MESSY subset (brand != generic)", messy),
                                    ("ALL support claims", triples)]:
            ra, rb = [], []
            for t in subset:
                qA = make_q(t, t["surface"])                              # raw brand
                qB = make_q(t, normalize(t["surface"], brand2generic))    # normalized
                ra.append(rank_of_gold(qA, idf, doc_vecs, docs, t["set_id"]))
                rb.append(rank_of_gold(qB, idf, doc_vecs, docs, t["set_id"]))
            A, B = metrics(ra), metrics(rb)
            print(f"{subset_name:34} {'A raw (brand)':18} {A['n']:>4} "
                  f"{A['R@1']:>6.3f} {A['R@3']:>6.3f} {A['MRR']:>6.3f}")
            print(f"{'':34} {'B OMOP-normalized':18} {B['n']:>4} "
                  f"{B['R@1']:>6.3f} {B['R@3']:>6.3f} {B['MRR']:>6.3f}")
            if A['n']:
                print(f"{'':34} {'>> lift (B-A)':18} {'':>4} "
                      f"{B['R@1']-A['R@1']:>+6.3f} {B['R@3']-A['R@3']:>+6.3f} "
                      f"{B['MRR']-A['MRR']:>+6.3f}")
            print()


if __name__ == "__main__":
    main()
