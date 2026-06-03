"""
protos.py  -  the shared nearest-prototype kernel for train.py and evaluate.py.
================================================================================
The "classifier" in this project is NEAREST-PROTOTYPE:
    prototype[intent] = L2-normalized MEAN of that intent's TRAIN embeddings,
and a query is labelled by its most-similar prototype (cosine).

This 3-line kernel used to be copy-pasted in two places — train.py's
`proto_recall1` (tracks test Recall@1 each epoch) and evaluate.py's
`build_prototypes` (the canonical evaluation). Duplication meant the two could
silently drift: change the centroid in one place and the training-time vs
eval-time numbers would quietly start measuring different things. It now lives
here so BOTH import one definition.

Why a separate file (and not just import evaluate.py from train.py):
  train.py puts customer_intent_search/ at the FRONT of sys.path to reuse its
  model.py + tokenizer.py. That dir ALSO has an evaluate.py (which imports
  sentence_transformers, unavailable here), so a bare `import evaluate` from
  train.py would grab the WRONG file and crash. This helper sidesteps that: it
  is dependency-light (numpy only — `model.encode` is passed in) and is NOT
  named `evaluate`, so nothing on sys.path shadows it.
"""
import numpy as np


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
