"""Whole-label NLI gate (Table 3, "Dense + whole-label NLI"; Equation 2 with
E(x) = D_K(x)).

  retrieve : dense bi-encoder, top-K labels by cosine similarity to the claim
  gate     : cross-encoder NLI with premise = the label's first 1,500 characters
             and hypothesis = the whole claim

SUPPORT iff the best entailment probability over the K labels reaches the
threshold, citing that label; otherwise FLAG. The script also reports retrieval
recall@K and a sweep over the threshold.

Run: python nli_gate_experiment.py [benchmark.jsonl]
"""
import json
import sys
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

RAW = "data/labels_raw.json"
BENCH = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark_ner.jsonl"
RETRIEVER = "sentence-transformers/all-MiniLM-L6-v2"
NLI = "cross-encoder/nli-deberta-v3-base"              # outputs contradiction/entailment/neutral
TOP_K = 5
THRESHOLD = 0.5          # entailment-prob cutoff for SUPPORT


def build_corpus():
    docs = []
    for r in json.load(open(RAW)):
        of = r.get("openfda", {})
        g = (of.get("generic_name") or [None])[0]
        if not g:
            continue
        ind = (r.get("indications_and_usage") or [""])[0]
        dos = (r.get("dosage_and_administration") or [""])[0]
        docs.append({"set_id": of.get("spl_set_id"), "drug": g.lower(),
                     "text": f"{g}. {ind} {dos}"[:1500]})
    return docs


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    docs = build_corpus()
    claims = [json.loads(l) for l in open(BENCH)]
    print(f"corpus passages: {len(docs)} | claims: {len(claims)}", file=sys.stderr)

    print("loading retriever + NLI models (first run downloads them)...", file=sys.stderr)
    bi = SentenceTransformer(RETRIEVER)
    ce = CrossEncoder(NLI)
    # locate the 'entailment' column in the NLI output
    id2label = ce.model.config.id2label
    ent_idx = next(i for i, l in id2label.items() if l.lower() == "entailment")

    # embed corpus + claims
    doc_emb = bi.encode([d["text"] for d in docs], normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)
    cl_emb = bi.encode([c["claim_text"] for c in claims], normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)

    # retrieve top-K per claim
    sims = cl_emb @ doc_emb.T                       # (n_claims, n_docs)
    topk = np.argsort(-sims, axis=1)[:, :TOP_K]

    # NLI on every (claim, retrieved-passage) pair, batched
    pairs, owner = [], []
    for ci, c in enumerate(claims):
        for di in topk[ci]:
            pairs.append([docs[di]["text"], c["claim_text"]])   # premise, hypothesis
            owner.append((ci, int(di)))
    logits = ce.predict(pairs, batch_size=32, show_progress_bar=True,
                        convert_to_numpy=True, apply_softmax=False)
    ent_prob = softmax(logits)[:, ent_idx]

    # best entailing label per claim, and whether the source label was retrieved
    best = {}
    for (ci, di), p in zip(owner, ent_prob):
        if ci not in best or p > best[ci][1]:
            best[ci] = (di, float(p))
    gold_in_topk = []
    for ci, c in enumerate(claims):
        if c["label"] != "support":
            continue
        gids = [docs[int(di)]["set_id"] for di in topk[ci]]
        gold_in_topk.append(c["gold_citation"]["set_id"] in gids)

    print(f"\nretriever : {RETRIEVER}")
    print(f"NLI gate  : {NLI}  (top-{TOP_K})\n")

    # --- retrieval recall ---------------------------------------------------
    rr = sum(gold_in_topk) / max(len(gold_in_topk), 1)
    print(f"RETRIEVAL recall@{TOP_K} (gold label among retrieved, support claims): "
          f"{rr:.3f}")
    print()

    # --- threshold sweep ----------------------------------------------------
    print("threshold sweep:")
    print(f"  {'thr':>4} {'TRUE-kept':>10} {'catch-rate':>11} {'overall-acc':>12} {'cite-prec':>10}")
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        per = defaultdict(lambda: [0, 0])
        ca = cc = 0
        for ci, c in enumerate(claims):
            di, p = best[ci]
            pred = "support" if p >= thr else "refute"
            key = c["perturbation"] or "TRUE"
            per[key][1] += 1
            per[key][0] += (pred == c["label"])
            if pred == "support":
                ca += 1
                cc += (docs[di]["set_id"] == c["gold_citation"]["set_id"])
        true_kept = per["TRUE"][0] / per["TRUE"][1]
        caught = sum(per[k][0] for k in per if k != "TRUE")
        tot_neg = sum(per[k][1] for k in per if k != "TRUE")
        overall = (per["TRUE"][0] + caught) / len(claims)
        print(f"  {thr:>4.1f} {true_kept:>10.3f} {caught/tot_neg:>11.3f} "
              f"{overall:>12.3f} {cc/max(ca,1):>10.3f}")


if __name__ == "__main__":
    main()
