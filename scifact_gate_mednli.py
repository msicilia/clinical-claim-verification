"""External validation on SciFact (Table 4, first row), with the biomedical NLI model (PubMedBERT-MNLI-MedNLI).

SciFact (Wadden et al., 2020) provides scientific claims, a cited abstract and a
SUPPORT / CONTRADICT / NOINFO verdict: atomic claims with explicit evidence, the
setting the gate is designed for.

For each dev claim with a cited document, we run the same gate as on the synthetic
benchmark: sentence-level NLI between each abstract sentence (premise) and the claim
(hypothesis), taking the max entailment and max contradiction probability. We report
3-way label accuracy and binary SUPPORT-vs-rest F1.

SciFact is expected unpacked under data/data/ (see the README).

Run: python scifact_gate_mednli.py
"""
import json, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sentence_transformers import CrossEncoder

BASE = "data/data/"
NLI = "pritamdeka/PubMedBERT-MNLI-MedNLI"


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def main():
    corpus = {json.loads(l)["doc_id"]: json.loads(l) for l in open(BASE + "corpus.jsonl")}
    claims = [json.loads(l) for l in open(BASE + "claims_dev.jsonl")]

    items = []   # (claim_text, abstract_sentences, gold_label)
    for c in claims:
        if not c.get("cited_doc_ids"):
            continue
        doc = corpus.get(c["cited_doc_ids"][0])
        if not doc:
            continue
        ev = c.get("evidence", {})
        labels = {e["label"] for evs in ev.values() for e in evs}
        gold = "SUPPORT" if "SUPPORT" in labels else \
               "CONTRADICT" if "CONTRADICT" in labels else "NOINFO"
        items.append((c["claim"], doc["abstract"], gold))
    print(f"SciFact dev items with cited evidence: {len(items)}", file=sys.stderr)

    ce = CrossEncoder(NLI)
    id2 = ce.model.config.id2label
    ent_i = next(i for i, l in id2.items() if l.lower() == "entailment")
    con_i = next(i for i, l in id2.items() if l.lower() == "contradiction")

    pairs, owner = [], []
    for k, (claim, sents, _) in enumerate(items):
        for s in sents[:12]:
            pairs.append([s, claim]); owner.append(k)
    logits = ce.predict(pairs, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, apply_softmax=False)
    prob = softmax(logits)
    max_ent = np.zeros(len(items)); max_con = np.zeros(len(items))
    for (k), e, c in zip(owner, prob[:, ent_i], prob[:, con_i]):
        max_ent[k] = max(max_ent[k], e); max_con[k] = max(max_con[k], c)

    from collections import Counter
    gold = [g for _, _, g in items]
    print(f"\ngate: {NLI} sentence-level  (gold dist: {dict(Counter(gold))})\n")
    print(f"  {'thr':>4} {'3way-acc':>9} {'SUP-P':>6} {'SUP-R':>6} {'SUP-F1':>7}")
    for thr in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = fp = fn = 0; correct3 = 0
        for k, g in enumerate(gold):
            if max_ent[k] >= thr and max_ent[k] >= max_con[k]:
                pred = "SUPPORT"
            elif max_con[k] >= thr and max_con[k] > max_ent[k]:
                pred = "CONTRADICT"
            else:
                pred = "NOINFO"
            correct3 += (pred == g)
            # binary SUPPORT vs rest
            tp += (pred == "SUPPORT" and g == "SUPPORT")
            fp += (pred == "SUPPORT" and g != "SUPPORT")
            fn += (pred != "SUPPORT" and g == "SUPPORT")
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn) if tp + fn else 0
        F = 2 * P * R / (P + R) if P + R else 0
        print(f"  {thr:>4.1f} {correct3/len(items):>9.3f} {P:>6.3f} {R:>6.3f} {F:>7.3f}")


if __name__ == "__main__":
    main()
