"""MedHallu with answer decomposition and a contradiction signal, using the general NLI model (DeBERTa)
(Section 4.4).

medhallu_experiment.py tests each answer as a whole. Here each answer is decomposed
and checked against the knowledge text; for each answer we:
  - split it into atomic sentences (sub-claims),
  - for each sub-claim, over the knowledge sentences take max entailment and max
    contradiction,
  - aggregate per answer: ent_mean / ent_min (weakest-supported sub-claim) and
    con_max (most-contradicted sub-claim),
and evaluate several decision rules. A hallucinated answer should have a poorly
supported or contradicted sub-claim.

Subsampled to the first N items for runtime.

Run: python medhallu_experiment_v2.py [N]
"""
import re
import sys
import warnings

import numpy as np
warnings.filterwarnings("ignore")
from datasets import load_dataset
from sentence_transformers import CrossEncoder

NLI = "cross-encoder/nli-deberta-v3-base"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MAX_K, MAX_A = 5, 3
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def sents(text, cap):
    s = [x.strip() for x in SENT_SPLIT.split(text) if 15 <= len(x.strip()) <= 400]
    return (s or [text[:300]])[:cap]


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled", split="train").select(range(N))
    ce = CrossEncoder(NLI)
    id2 = ce.model.config.id2label
    ent_i = next(i for i, l in id2.items() if l.lower() == "entailment")
    con_i = next(i for i, l in id2.items() if l.lower() == "contradiction")

    # instance = one (answer) with decomposed sub-claims
    instances, pairs, owner = [], [], []   # owner: (instance_idx, subclaim_idx)
    for r in ds:
        know = r["Knowledge"]
        know = " ".join(know) if isinstance(know, list) else str(know)
        ks = sents(know, MAX_K)
        diff = (r.get("Difficulty Level") or "na").lower()
        for ans, gold in [(r["Ground Truth"], 0), (r["Hallucinated Answer"], 1)]:
            asents = sents(str(ans), MAX_A)
            inst = {"gold": gold, "diff": diff, "n_sub": len(asents)}
            ii = len(instances)
            instances.append(inst)
            for si, a in enumerate(asents):
                for k in ks:
                    pairs.append([k, a])           # premise=knowledge, hypothesis=sub-claim
                    owner.append((ii, si))
    print(f"items: {len(ds)} | instances: {len(instances)} | NLI pairs: {len(pairs)}",
          file=sys.stderr)

    logits = ce.predict(pairs, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, apply_softmax=False)
    prob = softmax(logits)
    ent, con = prob[:, ent_i], prob[:, con_i]

    # aggregate per (instance, subclaim): max over knowledge sentences
    sub_ent, sub_con = {}, {}
    for (ii, si), e, c in zip(owner, ent, con):
        sub_ent[(ii, si)] = max(sub_ent.get((ii, si), 0.0), e)
        sub_con[(ii, si)] = max(sub_con.get((ii, si), 0.0), c)
    # aggregate per instance
    for k, inst in enumerate(instances):
        es = [sub_ent[(k, si)] for si in range(inst["n_sub"])]
        cs = [sub_con[(k, si)] for si in range(inst["n_sub"])]
        inst["ent_mean"], inst["ent_min"] = float(np.mean(es)), float(np.min(es))
        inst["con_max"] = float(np.max(cs))

    diffs = sorted({x["diff"] for x in instances})

    def report(name, score_fn, hallu_if_high, grid):
        best = None
        for thr in grid:
            tp = fp = fn = tn = 0
            for inst in instances:
                s = score_fn(inst)
                pred = 1 if (s > thr) == hallu_if_high else 0
                g = inst["gold"]
                tp += pred == 1 and g == 1; fp += pred == 1 and g == 0
                fn += pred == 0 and g == 1; tn += pred == 0 and g == 0
            acc = (tp + tn) / len(instances)
            pr = tp / (tp + fp) if tp + fp else 0
            rc = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
            if best is None or f1 > best[0]:
                best = (f1, thr, acc, pr, rc)
        f1, thr, acc, pr, rc = best
        print(f"  {name:28} best F1={f1:.3f} @thr={thr:.2f}  acc={acc:.3f} "
              f"P={pr:.3f} R={rc:.3f}")
        return best

    print(f"\nNLI gate: {NLI}  DECOMPOSED answer (<= {MAX_A} sub-claims, "
          f"<= {MAX_K} knowledge sentences)\n")
    g = [i / 20 for i in range(1, 20)]
    report("ent_mean low -> hallu", lambda x: x["ent_mean"], False, g)
    report("ent_min low -> hallu", lambda x: x["ent_min"], False, g)
    report("con_max high -> hallu", lambda x: x["con_max"], True, g)
    report("margin(ent_mean-con_max) low", lambda x: x["ent_mean"] - x["con_max"], False,
           [i / 20 for i in range(-19, 20)])


if __name__ == "__main__":
    main()
