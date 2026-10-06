"""Deterministic OMOP linker: map a surface string to a standard SNOMED Condition
concept using the OMOP CONCEPT and CONCEPT_SYNONYM vocabulary files.

link()   is concept(s) of Equation 1: exact concept-name match, then exact
         synonym match, case-insensitive; None when the string is not found.
expand() returns the concept and all its surface forms, syn(c) of Equation 1.

The index is built from two pre-filtered files (standard Condition concepts and
their synonyms) in data/omop/; the README gives the commands to produce them.
"""
import re

CONCEPT_COND = "data/omop/concept_cond.tsv"      # concept_id \t concept_name
SYNONYM_COND = "data/omop/synonym_cond.tsv"      # concept_id \t synonym_name


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


class OMOPConditionLinker:
    def __init__(self):
        self.cid2name = {}
        self.name2cid = {}     # exact concept_name (normalized) -> cid
        self.syn2cid = {}      # synonym (normalized) -> cid (first/standard wins)
        self.cid2syns = {}     # cid -> list of surface forms (name + synonyms)
        for line in open(CONCEPT_COND, encoding="utf-8"):
            cid, name = line.rstrip("\n").split("\t")
            self.cid2name[cid] = name
            self.name2cid.setdefault(_norm(name), cid)
            self.cid2syns.setdefault(cid, set()).add(name)
        for line in open(SYNONYM_COND, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            cid, syn = parts
            self.syn2cid.setdefault(_norm(syn), cid)
            self.cid2syns.setdefault(cid, set()).add(syn)

    def link(self, term):
        """Return (concept_id, standard_concept_name) or None.
        Deterministic: exact concept-name match first, then exact synonym match."""
        t = _norm(term)
        cid = self.name2cid.get(t) or self.syn2cid.get(t)
        if cid is None:
            return None
        return cid, self.cid2name[cid]

    def expand(self, term):
        """OMOP normalization as query expansion: map the surface form to its
        standard concept and return the concept's full set of surface forms
        (canonical name + all synonyms). Returns None if unlinkable."""
        hit = self.link(term)
        if hit is None:
            return None
        cid, name = hit
        return cid, name, sorted(self.cid2syns.get(cid, {name}))


if __name__ == "__main__":
    L = OMOPConditionLinker()
    print(f"loaded {len(L.cid2name)} condition concepts, {len(L.syn2cid)} synonyms")
    for t in ["heart attack", "high blood pressure", "high blood sugar",
              "kidney failure", "irregular heartbeat", "german measles",
              "low platelet count", "shortness of breath", "blood poisoning"]:
        print(f"  {t:24} -> {L.link(t)}")
