"""LLM verifier: a language model makes the verdict instead of the gate (Table 3,
last row; Section 4.2).

Everything except the verification step is shared with nli_gate_decomposed.py:

  claims     data/benchmark_ner.jsonl (2,515 claims)
  corpus     build_corpus() imported from nli_gate_decomposed
  retriever  the same model and the same top-K labels per claim
  metrics    TRUE-kept, per-perturbation catch-rate, overall catch-rate, cite-prec

The model receives the opening PASSAGE_CHARS characters of each retrieved label,
whereas the gate reads candidate sentences and the dose ceiling from the full
indications and dosage sections. It is told the same decision rule the gate
applies (drug indicated for the condition, and dose within the stated maximum) and
must name the passage it relies on, so that citation precision is comparable.

Requests use temperature 0 and Ollama's JSON output mode. Answers are cached per
(model, claim id) in data/llm_verifier_<model>.jsonl; a re-run resumes from the cache.

Run:
  python llm_verifier_baseline.py --model medgemma:4b --limit 30     # quick check
  python llm_verifier_baseline.py --model medgemma:4b                # all claims
  python llm_verifier_baseline.py --model qwen3:32b \\
      --classes TRUE,unsafe-dose --per-class 205 --seed 0            # 32B probe
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

# Imported, not copied, so the corpus cannot drift from the NLI-gate experiment.
from nli_gate_decomposed import build_corpus, RETRIEVER, TOP_K

BENCH = "data/benchmark_ner.jsonl"
OLLAMA = "http://localhost:11434/api/generate"
PASSAGE_CHARS = 1200
TIMEOUT = 180

PROMPT = """You are verifying a clinical claim against evidence taken from FDA drug labels.

CLAIM: {claim}

EVIDENCE PASSAGES:
{passages}

The claim is SUPPORTED only if the evidence shows both:
  (a) the drug is indicated for the stated condition, and
  (b) the stated dose does not exceed a maximum given in the evidence.
Otherwise the claim is REFUTED.

Reply with JSON only, no other text:
{{"verdict": "support" or "refute", "citation": <number of the passage you relied on, or null>}}"""

JSON_RE = re.compile(r"\{.*?\}", re.S)


def call(model, prompt):
    """Return the model's raw text, or None on transport failure."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 80},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode()).get("response", "")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"    transport error: {e}", file=sys.stderr)
        return None


def parse(text):
    """(verdict, citation) or (None, None) if the reply is unusable."""
    if not text:
        return None, None
    m = JSON_RE.search(text)
    if not m:
        return None, None
    try:
        o = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None
    v = str(o.get("verdict", "")).strip().lower()
    if v not in ("support", "refute"):
        return None, None
    c = o.get("citation")
    try:
        c = int(c)
    except (TypeError, ValueError):
        c = None
    return v, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all claims")
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--classes", default="",
                    help="comma-separated subset, e.g. TRUE,unsafe-dose")
    ap.add_argument("--per-class", type=int, default=0,
                    help="sample this many of each class (seeded, reproducible)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    docs = build_corpus()
    claims = [json.loads(l) for l in open(args.bench)]

    # Stratified probe: keep only the named classes, sampled reproducibly. Used to
    # test a larger model on the dose facet without querying all 2,515 claims. TRUE
    # claims are included as a control: a model that simply refuses more often would
    # "catch" more unsafe doses without any dose reasoning.
    if args.classes:
        want = [c.strip() for c in args.classes.split(",") if c.strip()]
        by_class = defaultdict(list)
        for c in claims:
            by_class[c.get("perturbation") or "TRUE"].append(c)
        rng = np.random.default_rng(args.seed)
        picked = []
        for k in want:
            pool = by_class.get(k, [])
            if args.per_class and len(pool) > args.per_class:
                idx = rng.choice(len(pool), size=args.per_class, replace=False)
                pool = [pool[i] for i in sorted(idx)]
            print(f"  class {k}: {len(pool)}", file=sys.stderr)
            picked.extend(pool)
        claims = picked
    if args.limit:
        claims = claims[:args.limit]
    print(f"corpus {len(docs)} | claims {len(claims)} | model {args.model}",
          file=sys.stderr)

    # --- retrieval, replicated exactly from nli_gate_decomposed -------------
    bi = SentenceTransformer(RETRIEVER)
    doc_emb = bi.encode([d["text"] for d in docs], normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)
    cl_emb = bi.encode([c["claim_text"] for c in claims], normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    topk = np.argsort(-(cl_emb @ doc_emb.T), axis=1)[:, :TOP_K]

    cache_path = f"data/llm_verifier_{args.model.replace(':', '_').replace('/', '_')}.jsonl"
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            try:
                r = json.loads(line)
                cache[r["id"]] = r
            except json.JSONDecodeError:
                pass
        print(f"cache: {len(cache)} answers in {cache_path}", file=sys.stderr)

    out = open(cache_path, "a")
    t0 = time.time()
    n_new = 0
    for ci, c in enumerate(claims):
        if c["id"] in cache:
            continue
        idxs = [int(d) for d in topk[ci]]
        passages = "\n".join(
            f"[{n + 1}] {docs[d]['text'][:PASSAGE_CHARS]}" for n, d in enumerate(idxs))
        prompt = PROMPT.format(claim=c["claim_text"], passages=passages)

        verdict, cit = parse(call(args.model, prompt))
        if verdict is None:                       # one retry, then give up
            verdict, cit = parse(call(args.model, prompt))
        rec = {"id": c["id"], "doc_idx": idxs, "verdict": verdict, "citation": cit}
        cache[c["id"]] = rec
        out.write(json.dumps(rec) + "\n")
        out.flush()
        n_new += 1
        if n_new % 10 == 0:
            el = time.time() - t0
            print(f"  {n_new} new in {el:.0f}s  ({el / n_new:.2f}s/claim)",
                  file=sys.stderr)
    out.close()

    # --- scoring, identical to the NLI-gate experiment ----------------------
    per = defaultdict(lambda: [0, 0])
    accepted = correct_cite = unparsed = 0
    for ci, c in enumerate(claims):
        r = cache.get(c["id"])
        if r is None or r["verdict"] is None:
            unparsed += 1
            pred, cited = "refute", None       # unusable reply counted as a refusal
        else:
            pred = r["verdict"]
            cn = r["citation"]
            cited = (docs[r["doc_idx"][cn - 1]]
                     if isinstance(cn, int) and 1 <= cn <= len(r["doc_idx"]) else None)
        key = c["perturbation"] or "TRUE"
        per[key][1] += 1
        per[key][0] += (pred == c["label"])
        if pred == "support":
            accepted += 1
            correct_cite += (cited is not None
                             and cited["set_id"] == c["gold_citation"]["set_id"])

    def acc(k):
        """Score for one class, or None when that class was not sampled -- printing
        0.000 for an unsampled class reads as a result rather than as a gap."""
        return per[k][0] / per[k][1] if per[k][1] else None

    def fmt(k):
        v = acc(k)
        return f"{v:.3f}" if v is not None else f"n/a (0 sampled)"

    caught = sum(per[k][0] for k in per if k != "TRUE")
    tot_neg = sum(per[k][1] for k in per if k != "TRUE")
    el = time.time() - t0

    print(f"\nmodel        : {args.model}   (temperature 0)")
    print(f"claims       : {len(claims)}   new queries: {n_new}   "
          f"{el / max(n_new, 1):.2f}s/claim")
    print(f"unusable     : {unparsed} ({unparsed / max(len(claims), 1):.1%}) "
          f"-- counted as refute")
    print(f"\n  TRUE-kept        {fmt('TRUE')}")
    print(f"  wrong-drug       {fmt('wrong-drug')}")
    print(f"  wrong-indication {fmt('wrong-indication')}")
    print(f"  unsafe-dose      {fmt('unsafe-dose')}")
    print(f"  catch-rate       {caught / max(tot_neg, 1):.3f}")
    print(f"  cite-prec        {correct_cite / max(accepted, 1):.3f}  "
          f"({accepted} accepted)")


if __name__ == "__main__":
    main()
