"""Entity extraction on CADEC: does the biomedical NER recover the adverse-event
span from the raw patient forum sentence (Section 4.4)? cadec_linker.py starts from
the annotated span; this script runs the extractor on the sentence itself.

CADEC (KevinSpaghetti/cadec) gives `text` (the forum sentence) and `ade` (the gold
adverse-event span, in lay language). We run the biomedical NER on `text` and ask:
did any predicted entity overlap the gold span? (overlap recall), and did it match
exactly? (exact recall). This measures extraction on real, messy, lay-language text
-- the realistic 'front door' of the pipeline.

Run: python cadec_extraction.py [N]
"""
import sys, warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from transformers import pipeline

SYMPTOM_TYPES = {"Disease_disorder", "Sign_symptom"}


def overlap(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    ner = pipeline("token-classification", model="d4data/biomedical-ner-all",
                   aggregation_strategy="simple")
    ds = load_dataset("KevinSpaghetti/cadec", split="train")

    seen, items = set(), []
    for r in ds:
        text, ade = (r.get("text") or "").strip(), (r.get("ade") or "").strip()
        if not text or not ade:
            continue
        gs = text.lower().find(ade.lower())
        if gs < 0 or (text, ade) in seen:    # span must be locatable in the sentence
            continue
        seen.add((text, ade))
        items.append((text, ade, gs, gs + len(ade)))
        if len(items) >= N:
            break
    print(f"CADEC sentences with a locatable gold span: {len(items)}", file=sys.stderr)

    n_any = n_overlap = n_overlap_sym = n_exact = 0
    multiword = [it for it in items if len(it[1].split()) >= 2]
    mw_overlap = 0
    for text, ade, g0, g1 in items:
        ents = ner(text)
        n_any += bool(ents)
        hit = hit_sym = exact = False
        for e in ents:
            s, t = e.get("start"), e.get("end")
            if s is None:
                continue
            if overlap(s, t, g0, g1):
                hit = True
                if e["entity_group"] in SYMPTOM_TYPES:
                    hit_sym = True
                if abs(s - g0) <= 2 and abs(t - g1) <= 2:
                    exact = True
        n_overlap += hit; n_overlap_sym += hit_sym; n_exact += exact
        if hit and len(ade.split()) >= 2:
            mw_overlap += 1

    n = len(items)
    print(f"\nEXTRACTION on real patient free text (biomedical NER, d4data):")
    print(f"  found any entity in the sentence        : {n_any/n:.3f}")
    print(f"  OVERLAP recall (any entity hits gold)   : {n_overlap/n:.3f}")
    print(f"  overlap recall as a symptom/disease type: {n_overlap_sym/n:.3f}")
    print(f"  EXACT-span recall (±2 chars)            : {n_exact/n:.3f}")
    print(f"  overlap recall on multi-word lay spans  : {mw_overlap/max(len(multiword),1):.3f}"
          f"  (n={len(multiword)})")


if __name__ == "__main__":
    main()
