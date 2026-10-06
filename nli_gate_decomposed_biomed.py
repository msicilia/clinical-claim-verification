"""Decomposed gate with a biomedical retriever (S-PubMedBert-MS-MARCO; Section 4.3).
Identical to nli_gate_decomposed.py except for the retriever.

A clinical claim is a conjunction: (drug indicated for condition) and (dose within
the label's ceiling). Each facet is checked separately:

  relation : NLI entailment of the hypothesis "<drug> is indicated for <condition>"
             against the candidate sentences P(x) of the top-K labels; p* is the
             best-scoring sentence and L* the label containing it
  dose     : numeric comparison of the claim's dose with the ceiling of L*, the
             largest dose stated in L*'s dosage section (mg, mcg and g are
             normalized to mg; other units and claims without a dose pass)

SUPPORT iff the relation is entailed and the dose does not exceed the ceiling;
otherwise FLAG. p* is the citation. Wrong-drug and wrong-indication claims fail the
relation check; unsafe-dose claims pass it and fail the dose check.

Run: python nli_gate_decomposed_biomed.py [benchmark.jsonl]
"""
import json
import re
import sys
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

RAW = "data/labels_raw.json"
BENCH = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark_ner.jsonl"
RETRIEVER = "pritamdeka/S-PubMedBert-MS-MARCO"
NLI = "cross-encoder/nli-deberta-v3-base"
TOP_K = 3
MAX_SENT = 6
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MAX_RE = re.compile(r"(?:maximum|not to exceed|up to)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(mg|mcg|g)", re.I)
DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b", re.I)
_MG = {"mg": 1.0, "mcg": 0.001, "g": 1000.0}


def to_mg(amount, unit):
    return amount * _MG.get(unit.lower(), 0.0)


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
        # dose ceiling in mg, from the full (untruncated) dosage section: the
        # largest of any explicit "maximum X" and any dose amount mentioned.
        ceil_mg = 0.0
        for amt, unit in list(DOSE_RE.findall(dos)) + list(MAX_RE.findall(dos)):
            ceil_mg = max(ceil_mg, to_mg(float(amt), unit))
        docs.append({"set_id": of.get("spl_set_id"), "drug": g.lower(),
                     "text": full[:1500], "sents": sents or [full[:300]], "ceil_mg": ceil_mg})
    return docs


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def candidate_sentences(claim, passages):
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
    if not scored:
        di, doc = passages[0]
        return [(di, doc["sents"][0])]
    return [(di, s) for _, di, s in scored[:MAX_SENT]]


def dose_violation(claim, doc):
    """True if the claim's dose exceeds the ceiling of doc, the label L* selected by
    the relation check rather than the top-ranked label, whose ranking an inflated
    dose can distort. Ceiling = the largest dose stated in that label's dosage
    section, including any explicit 'maximum X', normalized to mg."""
    dose = claim.get("dose") or {}
    amt, unit = dose.get("amount"), (dose.get("unit") or "").lower()
    if not amt or unit not in ("mg", "mcg", "g") or doc is None:
        return False
    ceil_mg = doc["ceil_mg"]
    return ceil_mg > 0.0 and to_mg(amt, unit) > ceil_mg


def main():
    docs = build_corpus()
    claims = [json.loads(l) for l in open(BENCH)]
    print(f"corpus: {len(docs)} | claims: {len(claims)} | bench: {BENCH}", file=sys.stderr)

    bi = SentenceTransformer(RETRIEVER)
    ce = CrossEncoder(NLI)
    ent_idx = next(i for i, l in ce.model.config.id2label.items() if l.lower() == "entailment")

    doc_emb = bi.encode([d["text"] for d in docs], normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)
    cl_emb = bi.encode([c["claim_text"] for c in claims], normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    topk = np.argsort(-(cl_emb @ doc_emb.T), axis=1)[:, :TOP_K]

    # facet-1 hypothesis = "<drug> is indicated for <condition>"  (no dose)
    pairs, owner = [], []
    for ci, c in enumerate(claims):
        passages = [(int(di), docs[int(di)]) for di in topk[ci]]
        hyp = f"{c['drug']['surface']} is indicated for {c['condition']['surface']}."
        for di, sent in candidate_sentences(c, passages):
            pairs.append([sent, hyp])
            owner.append((ci, di))
    print(f"NLI pairs: {len(pairs)}", file=sys.stderr)

    logits = ce.predict(pairs, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, apply_softmax=False)
    ent = softmax(logits)[:, ent_idx]
    best = {}
    for (ci, di), p in zip(owner, ent):
        if ci not in best or p > best[ci][1]:
            best[ci] = (di, float(p))

    print(f"\nretriever : {RETRIEVER}")
    print(f"NLI gate  : {NLI}  DECOMPOSED (facet-1 entailment + facet-2 dose check)\n")
    print("threshold sweep (facet-1 entailment threshold):")
    print(f"  {'thr':>4} {'TRUE-kept':>10} {'wrong-drug':>11} {'wrong-ind':>10} "
          f"{'unsafe-dose':>12} {'catch':>7} {'cite-prec':>10}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        per = defaultdict(lambda: [0, 0]); ca = cc = 0
        for ci, c in enumerate(claims):
            di, p = best.get(ci, (-1, 0.0))
            ent_ok = p >= thr
            cited = docs[di] if di >= 0 else None
            pred = "support" if (ent_ok and not dose_violation(c, cited)) else "refute"
            key = c["perturbation"] or "TRUE"
            per[key][1] += 1; per[key][0] += (pred == c["label"])
            if pred == "support":
                ca += 1
                cc += (di >= 0 and docs[di]["set_id"] == c["gold_citation"]["set_id"])
        def acc(k): return per[k][0] / per[k][1] if per[k][1] else 0.0
        caught = sum(per[k][0] for k in per if k != "TRUE")
        tot_neg = sum(per[k][1] for k in per if k != "TRUE")
        print(f"  {thr:>4.1f} {acc('TRUE'):>10.3f} {acc('wrong-drug'):>11.3f} "
              f"{acc('wrong-indication'):>10.3f} {acc('unsafe-dose'):>12.3f} "
              f"{caught/tot_neg:>7.3f} {cc/max(ca,1):>10.3f}")


if __name__ == "__main__":
    main()
