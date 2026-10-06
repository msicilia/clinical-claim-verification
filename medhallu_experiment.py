"""External validation on MedHallu, whole-answer entailment (Table 4, third row).

MedHallu (Pandit et al., EMNLP 2025) provides medical hallucinations generated from PubMedQA, with both
a correct ("Ground Truth") and a "Hallucinated Answer" per question plus the source
"Knowledge". We test whether the same sentence-level entailment gate used in the
synthetic study generalizes to this external task.

Setup. For each MedHallu item we form two instances:
  (Knowledge, Ground Truth)        -> label = supported  (not hallucinated)
  (Knowledge, Hallucinated Answer) -> label = hallucinated
Gate. Split Knowledge into sentences; NLI(premise=best knowledge sentence,
hypothesis=answer); predict HALLUCINATED iff the max entailment prob is below a
threshold. Report detection accuracy / precision / recall / F1, by difficulty.

Run: python medhallu_experiment.py
"""
import re
import sys
import warnings

import numpy as np
warnings.filterwarnings("ignore")
from datasets import load_dataset
from sentence_transformers import CrossEncoder

NLI = "cross-encoder/nli-deberta-v3-base"
MAX_SENT = 6
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def sents(text):
    s = [x.strip() for x in SENT_SPLIT.split(text) if 15 <= len(x.strip()) <= 400]
    return s or [text[:300]]


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled", split="train")
    print(f"MedHallu pqa_labeled: {len(ds)} items", file=sys.stderr)

    ce = CrossEncoder(NLI)
    ent_idx = next(i for i, l in ce.model.config.id2label.items() if l.lower() == "entailment")

    # build instances + candidate (knowledge-sentence, answer) NLI pairs
    instances, pairs, owner = [], [], []
    for i, r in enumerate(ds):
        know = r["Knowledge"]
        know = " ".join(know) if isinstance(know, list) else str(know)
        ks = sents(know)[:MAX_SENT]
        diff = (r.get("Difficulty Level") or "na").lower()
        for ans, gold in [(r["Ground Truth"], 0), (r["Hallucinated Answer"], 1)]:
            idx = len(instances)
            instances.append({"gold": gold, "diff": diff})
            hyp = str(ans)[:400]
            for s in ks:
                pairs.append([s, hyp])
                owner.append(idx)
    print(f"instances: {len(instances)} | NLI pairs: {len(pairs)}", file=sys.stderr)

    logits = ce.predict(pairs, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, apply_softmax=False)
    ent = softmax(logits)[:, ent_idx]
    best = np.zeros(len(instances))
    for o, p in zip(owner, ent):
        if p > best[o]:
            best[o] = p

    print(f"\nNLI gate: {NLI}  (sentence-level, <= {MAX_SENT} knowledge sentences)")
    print("predict HALLUCINATED iff max entailment prob < threshold\n")

    diffs = sorted({x["diff"] for x in instances})
    print(f"  {'thr':>4} {'acc':>6} {'precision':>10} {'recall':>8} {'F1':>6}   "
          + "  ".join(f"F1[{d}]" for d in diffs))
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        tp = fp = fn = tn = 0
        perdiff = {d: [0, 0, 0] for d in diffs}  # tp, fp, fn
        for k, inst in enumerate(instances):
            pred = 1 if best[k] < thr else 0          # 1 = hallucinated
            g = inst["gold"]
            tp += (pred == 1 and g == 1); fp += (pred == 1 and g == 0)
            fn += (pred == 0 and g == 1); tn += (pred == 0 and g == 0)
            d = inst["diff"]
            perdiff[d][0] += (pred == 1 and g == 1)
            perdiff[d][1] += (pred == 1 and g == 0)
            perdiff[d][2] += (pred == 0 and g == 1)
        acc = (tp + tn) / len(instances)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1d = []
        for d in diffs:
            t, p_, n = perdiff[d]
            pr = t / (t + p_) if t + p_ else 0.0
            rc = t / (t + n) if t + n else 0.0
            f1d.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        print(f"  {thr:>4.1f} {acc:>6.3f} {prec:>10.3f} {rec:>8.3f} {f1:>6.3f}   "
              + "  ".join(f"{x:>6.3f}" for x in f1d))


if __name__ == "__main__":
    main()
