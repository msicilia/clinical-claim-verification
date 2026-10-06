"""C1 with exact OMOP lookup plus an embedding fallback (Table 2, fallback row).

Terms not found by exact lookup are linked to the nearest OMOP standard condition
concept by embedding similarity. Every target is an OMOP standard SNOMED concept,
and the query is expanded with the linked concept's synonym set.

Arms: raw | naive-string | exact OMOP | exact + embedding fallback | oracle.

Requires the embeddings cached by embed_omop_conditions.py.
Run: python omop_c1_full.py
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer

from condition_experiment import (fetch_corpus, tfidf, evaluate, naive_normalize,
                                   SYNONYMS)
from omop_link import OMOPConditionLinker, _norm

ENC = "pritamdeka/S-PubMedBert-MS-MARCO"


def main():
    docs = fetch_corpus()
    idf, doc_vecs = tfidf(docs)
    conds = sorted(set(d["condition"] for d in docs))
    lay_of = {v: k for k, v in SYNONYMS.items()}

    L = OMOPConditionLinker()
    cond_emb = np.load("data/omop/cond_emb.npy")
    cond_ids = open("data/omop/cond_ids.txt").read().split("\n")
    enc = SentenceTransformer(ENC)
    print(f"OMOP: {len(L.cid2name)} concepts; embedding fallback over "
          f"{len(cond_ids)} concise concepts\n")

    lays = [lay_of[c] for c in conds]
    lay_emb = enc.encode(lays, normalize_embeddings=True, convert_to_numpy=True)

    def full_link(lay, i):
        hit = L.link(lay)
        if hit:
            return hit[0], hit[1], "exact"
        j = int((lay_emb[i] @ cond_emb.T).argmax())
        cid = cond_ids[j]
        return cid, L.cid2name.get(cid, "?"), "embed"

    agg = {"raw": [0.0]*3, "naive-string": [0.0]*3, "exact OMOP": [0.0]*3,
           "OMOP exact+embedding": [0.0]*3, "oracle": [0.0]*3}
    n_exact = n_embed = 0
    print(f"{'lay term':24} {'OMOP link':32} {'via':>5} {'rawP5':>6} {'fullP5':>6}")
    print("-" * 82)
    for i, cond in enumerate(conds):
        lay = lays[i]
        # exact-only query
        ex = L.expand(lay)
        ex_query = " ".join(ex[2]) if ex else lay
        # full (exact + embed fallback)
        cid, name, via = full_link(lay, i)
        n_exact += via == "exact"; n_embed += via == "embed"
        full_query = " ".join(sorted(L.cid2syns.get(cid, {name})))

        a = evaluate(lay, idf, doc_vecs, docs, cond)
        b = evaluate(naive_normalize(lay), idf, doc_vecs, docs, cond)
        e = evaluate(ex_query, idf, doc_vecs, docs, cond)
        f = evaluate(full_query, idf, doc_vecs, docs, cond)
        o = evaluate(cond, idf, doc_vecs, docs, cond)
        for arm, v in zip(agg, (a, b, e, f, o)):
            for k in range(3):
                agg[arm][k] += v[k]
        print(f"{lay:24} {name[:32]:32} {via:>5} {a[0]:>6.2f} {f[0]:>6.2f}")
    n = len(conds)
    print("-" * 82)
    print(f"linking: {n_exact} exact + {n_embed} embedding-fallback = {n}/{n} covered\n")
    print(f"{'arm':26} {'P@5':>7} {'R@10':>7} {'MRR':>7}")
    for arm, (P, R, M) in agg.items():
        print(f"{arm:26} {P/n:>7.3f} {R/n:>7.3f} {M/n:>7.3f}")


if __name__ == "__main__":
    main()
