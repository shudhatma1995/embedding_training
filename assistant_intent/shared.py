"""
shared.py  -  helpers used by BOTH train.py and evaluate.py.
================================================================================
Two small things lived in duplicate before this module existed:

  load_grouped     - read the {text,intent} json into {intent: [texts]}.
                     train.py and evaluate.py each carried an identical copy.
  build_prototypes - the nearest-prototype kernel (encode → mean → L2-normalize).
                     train.py uses it to track test Recall@1 each epoch; evaluate.py
                     uses it as the canonical classifier. Duplication risked silent
                     drift between the training-time and eval-time numbers.

Keeping both here gives ONE source of truth. The module is deliberately
dependency-light (json + numpy only — `model.encode` is passed in) and is NOT
named `evaluate`, so nothing on sys.path can shadow it (train.py puts
customer_intent_search/ — which has its own sentence_transformers-importing
evaluate.py — at the front of sys.path).
"""
import json

import numpy as np


def load_grouped(path: str, keep: list) -> dict:
    """Load [{text,intent}] json → {intent: [texts]} for the kept intents.
    Keys follow `keep` order; an intent with no rows maps to an empty list."""
    with open(path) as f:
        rows = json.load(f)
    groups = {k: [] for k in keep}
    for r in rows:
        if r["intent"] in keep:
            groups[r["intent"]].append(r["text"])
    return groups


def build_prototypes(model, tok, train_groups, device):
    """prototype[intent] = L2-normalized MEAN of that intent's train embeddings.

    Returns (protos, intents):
        protos  [C, D]  one unit vector per intent (the centroids)
        intents [C]     intent names, in train_groups insertion order
    Averaging unit vectors shrinks the length, so we re-normalize (1e-9 guards
    the divide-by-zero if an intent somehow had no examples).
    """
    intents = list(train_groups.keys())
    protos = []
    for it in intents:
        e = model.encode(train_groups[it], tok, device=device)   # [n,D], L2-normed
        m = e.mean(axis=0)                                        # centroid
        protos.append(m / (np.linalg.norm(m) + 1e-9))            # back to unit length
    return np.stack(protos), intents                             # [C,D], [C]
