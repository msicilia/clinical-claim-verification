"""C1 with an LLM-assisted OMOP linker (Table 2, LLM-assisted row).

Only a third of the lay terms are exact OMOP synonyms. A local medical LLM
(medgemma:4b via Ollama) proposes a standard clinical term for each lay term, and
that term is then linked deterministically to an OMOP standard SNOMED concept: the
model never produces a concept identifier, and the vocabulary remains the source
of truth. The query is expanded with the linked concept's synonym set.

Arms: raw | OMOP (exact) | LLM+OMOP | oracle.
LLM answers are requested at temperature 0 and cached in data/llm_links.json.

Run: python llm_omop_c1.py          (requires `ollama serve` with medgemma:4b)
"""
import json, os, urllib.request, warnings
warnings.filterwarnings("ignore")

from condition_experiment import fetch_corpus, tfidf, evaluate, SYNONYMS
from omop_link import OMOPConditionLinker

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "medgemma:4b"
CACHE = "data/llm_links.json"
PROMPT = ("What is the single standard SNOMED clinical term for the lay phrase "
          "'{term}'? Reply with ONLY the medical term, nothing else.")


def llm_term(term, cache):
    if term in cache:
        return cache[term]
    body = json.dumps({"model": MODEL, "stream": False, "options": {"temperature": 0},
                       "prompt": PROMPT.format(term=term)}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=120)).get("response", "").strip()
    resp = resp.splitlines()[0].strip().strip(".") if resp else term
    cache[term] = resp
    return resp


def main():
    docs = fetch_corpus()
    idf, doc_vecs = tfidf(docs)
    conds = sorted(set(d["condition"] for d in docs))
    lay_of = {v: k for k, v in SYNONYMS.items()}
    L = OMOPConditionLinker()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    agg = {"raw": [0.0]*3, "OMOP (exact)": [0.0]*3,
           "LLM+OMOP": [0.0]*3, "oracle": [0.0]*3}
    n_link = 0
    print(f"{'lay term':22} {'LLM clinical term':26} {'OMOP grounding':26} {'raw':>5} {'llm':>5}")
    print("-" * 90)
    for cond in conds:
        lay = lay_of[cond]
        clinical = llm_term(lay, cache)
        hit = L.link(clinical)                      # ground the LLM term in OMOP
        if hit:
            n_link += 1
            cid, name = hit
            llm_query = " ".join(sorted(L.cid2syns.get(cid, {name})))
            ground = f"{name} ({cid})"
        else:
            llm_query = clinical                    # use LLM term directly if not in OMOP
            ground = "-- not in OMOP, use LLM term --"
        ex = L.expand(lay)
        ex_query = " ".join(ex[2]) if ex else lay
        a = evaluate(lay, idf, doc_vecs, docs, cond)
        e = evaluate(ex_query, idf, doc_vecs, docs, cond)
        m = evaluate(llm_query, idf, doc_vecs, docs, cond)
        o = evaluate(cond, idf, doc_vecs, docs, cond)
        for arm, v in zip(agg, (a, e, m, o)):
            for j in range(3):
                agg[arm][j] += v[j]
        print(f"{lay:22} {clinical[:26]:26} {ground[:26]:26} {a[0]:>5.2f} {m[0]:>5.2f}")
    json.dump(cache, open(CACHE, "w"), indent=0)
    n = len(conds)
    print("-" * 90)
    print(f"LLM->OMOP grounding rate: {n_link}/{n} = {n_link/n:.2f}\n")
    print(f"{'arm':22} {'P@5':>7} {'R@10':>7} {'MRR':>7}")
    for arm, (P, R, M) in agg.items():
        print(f"{arm:22} {P/n:>7.3f} {R/n:>7.3f} {M/n:>7.3f}")


if __name__ == "__main__":
    main()
