"""Sentence-level NLI gate (Table 3, "+ sentence-level NLI"; Equation 2 with
E(x) = P(x)).

Retrieve the top-K labels, split their indications and dosage sections into
sentences, and keep up to MAX_SENT sentences that mention the claim's drug or
condition, ranked by the number of such mentions (the candidates P(x)). Each is
scored by NLI against the whole claim; SUPPORT iff the best entailment probability
reaches the threshold, citing the label that contains that sentence.

Run: python nli_gate_sentence.py [benchmark.jsonl]
"""
import json
import re
import sys
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

RAW = "data/labels_raw.json"
BENCH = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark_ner.jsonl"
RETRIEVER = "sentence-transformers/all-MiniLM-L6-v2"
NLI = "cross-encoder/nli-deberta-v3-base"
TOP_K = 3
MAX_SENT = 6                       # candidate sentences scored per claim
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def build_corpus():
    docs = []
    for r in json.load(open(RAW)):
        of = r.get("openfda", {})
        g = (of.get("generic_name") or [None])[0]
        if not g:
            continue
        ind = (r.get("indications_and_usage") or [""])[0]
        dos = (r.get("dosage_and_administration") or [""])[0]
        full = f"{g}. {ind} {dos}"
        sents = [s.strip() for s in SENT_SPLIT.split(full) if 15 <= len(s.strip()) <= 400]
        docs.append({"set_id": of.get("spl_set_id"), "drug": g.lower(),
                     "text": full[:1500], "sents": sents or [full[:300]]})
    return docs


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def candidate_sentences(claim, passages):
    """Sentences from retrieved passages mentioning the drug or condition."""
    drug = claim["drug"]["surface"].split()[0].lower()
    cond_toks = {w for w in re.findall(r"[a-z]+", claim["condition"]["surface"].lower())
                 if len(w) > 3}
    scored = []
    for di, doc in passages:
        for s in doc["sents"]:
            sl = s.lower()
            hit = (drug in sl) + sum(1 for w in cond_toks if w in sl)
            if hit:
                scored.append((hit, di, s))
    scored.sort(key=lambda x: -x[0])
    if not scored:                                 # fallback: first sentence of top passage
        di, doc = passages[0]
        return [(di, doc["sents"][0])]
    return [(di, s) for _, di, s in scored[:MAX_SENT]]


def main():
    docs = build_corpus()
    claims = [json.loads(l) for l in open(BENCH)]
    print(f"corpus passages: {len(docs)} | claims: {len(claims)} | bench: {BENCH}",
          file=sys.stderr)

    bi = SentenceTransformer(RETRIEVER)
    ce = CrossEncoder(NLI)
    ent_idx = next(i for i, l in ce.model.config.id2label.items() if l.lower() == "entailment")

    doc_emb = bi.encode([d["text"] for d in docs], normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)
    cl_emb = bi.encode([c["claim_text"] for c in claims], normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    topk = np.argsort(-(cl_emb @ doc_emb.T), axis=1)[:, :TOP_K]

    # gather candidate (claim, sentence) pairs
    pairs, owner = [], []
    for ci, c in enumerate(claims):
        passages = [(int(di), docs[int(di)]) for di in topk[ci]]
        for di, sent in candidate_sentences(c, passages):
            pairs.append([sent, c["claim_text"]])
            owner.append((ci, di))
    print(f"NLI pairs (sentence-level): {len(pairs)}", file=sys.stderr)

    logits = ce.predict(pairs, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, apply_softmax=False)
    ent = softmax(logits)[:, ent_idx]

    best = {}
    for (ci, di), p in zip(owner, ent):
        if ci not in best or p > best[ci][1]:
            best[ci] = (di, float(p))

    # retrieval recall@K for reference
    rr = []
    for ci, c in enumerate(claims):
        if c["label"] != "support":
            continue
        gids = [docs[int(di)]["set_id"] for di in topk[ci]]
        rr.append(c["gold_citation"]["set_id"] in gids)

    print(f"\nretriever : {RETRIEVER}")
    print(f"NLI gate  : {NLI}  SENTENCE-LEVEL  (top-{TOP_K} passages, <= {MAX_SENT} sents)\n")
    print(f"retrieval recall@{TOP_K} (support claims): {sum(rr)/max(len(rr),1):.3f}\n")
    print("threshold sweep:")
    print(f"  {'thr':>4} {'TRUE-kept':>10} {'catch-rate':>11} {'overall-acc':>12} {'cite-prec':>10}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        per = defaultdict(lambda: [0, 0]); ca = cc = 0
        for ci, c in enumerate(claims):
            di, p = best.get(ci, (-1, 0.0))
            pred = "support" if p >= thr else "refute"
            key = c["perturbation"] or "TRUE"
            per[key][1] += 1; per[key][0] += (pred == c["label"])
            if pred == "support":
                ca += 1
                cc += (di >= 0 and docs[di]["set_id"] == c["gold_citation"]["set_id"])
        true_kept = per["TRUE"][0] / per["TRUE"][1]
        caught = sum(per[k][0] for k in per if k != "TRUE")
        tot_neg = sum(per[k][1] for k in per if k != "TRUE")
        overall = (per["TRUE"][0] + caught) / len(claims)
        print(f"  {thr:>4.1f} {true_kept:>10.3f} {caught/tot_neg:>11.3f} "
              f"{overall:>12.3f} {cc/max(ca,1):>10.3f}")


if __name__ == "__main__":
    main()
