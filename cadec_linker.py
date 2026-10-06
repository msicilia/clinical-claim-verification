"""External validation on CADEC: the LLM-assisted OMOP linker on real patient lay
language (Table 4, second row).

CADEC (KevinSpaghetti/cadec) is real patient forum text with adverse-event spans in
lay language (`ade`, e.g. "could barely walk across the room") and a gold clinical
preferred term (`term_PT`, e.g. "gait disturbance"). This tests the linker beyond the 30
lay terms of C1.

For each pair we link BOTH sides to an OMOP standard concept and check agreement:
  gold  = OMOP.link(term_PT)                          (gold clinical term -> concept)
  exact = OMOP.link(ade)                              (raw lay span, no LLM)
  llm   = OMOP.link( medgemma(ade) )                 (LLM bridges -> OMOP grounds)
Accuracy = fraction whose predicted concept_id matches the gold concept_id.

Run: python cadec_linker.py [N]
"""
import json, os, sys, urllib.request, warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from omop_link import OMOPConditionLinker

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "medgemma:4b"
CACHE = "data/cadec_llm.json"
PROMPT = ("What is the single standard SNOMED clinical term for this patient phrase: "
          "'{term}'? Reply with ONLY the medical term, nothing else.")


def llm_term(term, cache):
    if term in cache:
        return cache[term]
    body = json.dumps({"model": MODEL, "stream": False, "options": {"temperature": 0},
                       "prompt": PROMPT.format(term=term)}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120)).get("response", "").strip()
    r = r.splitlines()[0].strip().strip(".") if r else term
    cache[term] = r
    return r


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    L = OMOPConditionLinker()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    ds = load_dataset("KevinSpaghetti/cadec", split="train")
    seen, pairs = set(), []
    for r in ds:
        ade, pt = (r.get("ade") or "").strip(), (r.get("term_PT") or "").strip()
        if not ade or not pt or (ade, pt) in seen:
            continue
        gold = L.link(pt)                       # only keep pairs whose gold term is in OMOP
        if gold is None:
            continue
        seen.add((ade, pt))
        pairs.append((ade, pt, gold[0]))
        if len(pairs) >= N:
            break
    print(f"CADEC pairs with OMOP-linkable gold term: {len(pairs)}", file=sys.stderr)

    n_exact_ok = n_llm_ok = n_exact_cov = n_llm_cov = 0
    shown = 0
    print(f"{'lay span (ade)':38} {'gold PT':22} {'LLM->OMOP':22} {'ok':>3}")
    print("-" * 88)
    for ade, pt, gold_cid in pairs:
        ex = L.link(ade)
        n_exact_cov += ex is not None
        n_exact_ok += ex is not None and ex[0] == gold_cid
        clinical = llm_term(ade, cache)
        ll = L.link(clinical)
        n_llm_cov += ll is not None
        ok = ll is not None and ll[0] == gold_cid
        n_llm_ok += ok
        if shown < 22:
            shown += 1
            print(f"{ade[:38]:38} {pt[:22]:22} {(ll[1] if ll else '-')[:22]:22} "
                  f"{'OK' if ok else '':>3}")
    json.dump(cache, open(CACHE, "w"))
    n = len(pairs)
    print("-" * 88)
    print(f"\nlinking accuracy (predicted concept == gold concept) on real lay spans:")
    print(f"  raw lay span -> OMOP exact : coverage {n_exact_cov/n:.2f}, "
          f"accuracy {n_exact_ok/n:.3f}")
    print(f"  LLM + OMOP                 : coverage {n_llm_cov/n:.2f}, "
          f"accuracy {n_llm_ok/n:.3f}")


if __name__ == "__main__":
    main()
