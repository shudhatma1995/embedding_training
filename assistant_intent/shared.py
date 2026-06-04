"""
shared.py  -  the nearest-prototype classifier, used by BOTH train.py and evaluate.py.
================================================================================
This is the ONE source of truth for "how do we turn the embedder into a classifier":

  build_prototypes - the kernel: encode → mean → L2-normalize, one centroid/intent.
  proto_recall1    - the in-training quality signal: nearest-prototype Recall@1 on
                     the held-out test queries.

train.py calls proto_recall1 to track test Recall@1 each epoch; evaluate.py calls
build_prototypes directly as the canonical classifier. Keeping both here stops the
training-time and eval-time numbers from silently drifting apart.

Deliberately dependency-light (numpy + torch only — `model.encode` is passed in) and
NOT named `evaluate`, so nothing on sys.path can shadow it (train.py puts
customer_intent_search/ — which has its own sentence_transformers-importing
evaluate.py — at the front of sys.path).
"""
import numpy as np
import torch


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


@torch.no_grad()
def proto_recall1(model, tok, train_groups, test_groups, device) -> float:
    """
    Centroid classifier: prototype[intent] = L2-normalized MEAN of that intent's
    train embeddings. Each test query is labelled by its nearest prototype (cosine).
    Returns Recall@1 over the held-out test queries (the 4 real intents).

    This is the honest generalization signal: test.json uses unseen phrasings AND
    unseen entities, so this number reflects pattern-learning, not memorization.
    """
    model.eval()
    protos, intents = build_prototypes(model, tok, train_groups, device)  # [C,D], [C]

    correct = total = 0
    for ti, it in enumerate(intents):
        q = model.encode(test_groups[it], tok, device=device)    # [m,D]
        if len(q) == 0:
            continue
        pred = (q @ protos.T).argmax(axis=1)                     # nearest prototype
        correct += int((pred == ti).sum())
        total += len(q)
    model.train()
    return correct / max(1, total)
