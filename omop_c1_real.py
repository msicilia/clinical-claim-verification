"""C1 with exact OMOP linking (Table 2, exact-lookup row).

Each lay query is looked up deterministically in the OMOP CONCEPT /
CONCEPT_SYNONYM files: surface form -> standard SNOMED condition concept -> query
expanded with all surface forms of that concept (s' in Equation 1). Terms not found
fall back to the raw lay query.

Arms (same TF-IDF retriever and corpus as condition_experiment.py):
  raw                : lay term as written
  naive-string       : knowledge-free cleanup
  OMOP (exact)       : OMOP vocabulary lookup + concept synonym expansion
  oracle (hand table): perfect lay -> canonical mapping (upper bound)

Run: python omop_c1_real.py
"""
from condition_experiment import (fetch_corpus, tfidf, evaluate, naive_normalize,
                                   SYNONYMS)
from omop_link import OMOPConditionLinker


def main():
    docs = fetch_corpus()
    idf, doc_vecs = tfidf(docs)
    conds = sorted(set(d["condition"] for d in docs))
    lay_of = {v: k for k, v in SYNONYMS.items()}

    print("loading OMOP linker (OMOP CONCEPT + CONCEPT_SYNONYM files) ...")
    L = OMOPConditionLinker()
    print(f"  {len(L.cid2name)} condition concepts, {len(L.syn2cid)} synonyms\n")

    agg = {"raw": [0.0]*3, "naive-string": [0.0]*3,
           "OMOP (exact)": [0.0]*3, "oracle (hand table)": [0.0]*3}
    linked = 0
    print(f"{'lay term':26} {'OMOP link':34} {'rawP5':>6} {'omopP5':>6}")
    print("-" * 80)
    for cond in conds:
        lay = lay_of[cond]
        exp = L.expand(lay)
        if exp:
            linked += 1
            cid, name, syns = exp
            omop_query = " ".join(syns)              # concept-expanded query
            link_str = f"{name} ({cid})"
        else:
            omop_query = lay                          # fallback: no normalization
            link_str = "-- unlinked (fallback raw) --"
        a = evaluate(lay, idf, doc_vecs, docs, cond)
        b = evaluate(naive_normalize(lay), idf, doc_vecs, docs, cond)
        c = evaluate(omop_query, idf, doc_vecs, docs, cond)
        o = evaluate(cond, idf, doc_vecs, docs, cond)            # oracle = hand canonical
        for arm, v in zip(agg, (a, b, c, o)):
            for j in range(3):
                agg[arm][j] += v[j]
        print(f"{lay:26} {link_str[:34]:34} {a[0]:>6.2f} {c[0]:>6.2f}")
    n = len(conds)
    print("-" * 80)
    print(f"exact OMOP linking coverage: {linked}/{n} = {linked/n:.2f}\n")
    print(f"{'arm':24} {'P@5':>7} {'R@10':>7} {'MRR':>7}")
    for arm, (P, R, M) in agg.items():
        print(f"{arm:24} {P/n:>7.3f} {R/n:>7.3f} {M/n:>7.3f}")


if __name__ == "__main__":
    main()
