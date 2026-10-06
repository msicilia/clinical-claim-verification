"""Embed the OMOP standard condition concept names, for the embedding fallback of
omop_c1_full.py. Run once; caches data/omop/cond_emb.npy and cond_ids.txt.

Run: python embed_omop_conditions.py
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer

ENC = "pritamdeka/S-PubMedBert-MS-MARCO"
ids, names = [], []
for line in open("data/omop/concept_cond.tsv", encoding="utf-8"):
    cid, name = line.rstrip("\n").split("\t")
    # keep concise names (lay terms map to general concepts, not verbose ones)
    if len(name.split()) <= 5:
        ids.append(cid); names.append(name)
print(f"embedding {len(names)} concise condition concept names ...")
emb = SentenceTransformer(ENC).encode(names, normalize_embeddings=True,
                                      convert_to_numpy=True, batch_size=128,
                                      show_progress_bar=True)
np.save("data/omop/cond_emb.npy", emb)
open("data/omop/cond_ids.txt", "w").write("\n".join(ids))
print("cached cond_emb.npy + cond_ids.txt")
