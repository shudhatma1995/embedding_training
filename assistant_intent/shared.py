"""
shared.py  -  the nearest-prototype classifier kernel, encoder-agnostic.
================================================================================
The ONE source of truth for "turn an embedder into a classifier". Everything here
depends only on an ENCODER — a plain callable:

    encode(texts: list[str]) -> np.ndarray   # shape [N, D], L2-normalized rows

Our from-scratch model is adapted to that interface by `make_encoder`; a pretrained
model (e.g. sentence-transformers MiniLM) supplies its own one-line adapter. So the
exact same prototype + threshold evaluation runs against ANY encoder — which is what
makes an apples-to-apples baseline possible.

  make_encoder       - wrap our (model, tok, device) as an Encoder callable.
  mine_hard_negatives- encode all train queries → per-query top-k confusable
                       different-intent neighbours (the hard-negative pool).
  build_prototypes   - encode → mean → L2-normalize, one centroid per intent.
  proto_recall1      - in-training quality signal: nearest-prototype Recall@1 on test.

Deliberately dependency-light (numpy only) and NOT named `evaluate`, so nothing on
sys.path can shadow it (train.py puts customer_intent_search/ — which has its own
sentence_transformers-importing evaluate.py — at the front of sys.path).
"""

import numpy as np


def make_encoder(model, tok, device):
    """Adapt our from-scratch model to the Encoder interface: texts -> [N, D]
    L2-normalized ndarray. model.encode already runs under eval()+no_grad()."""
    return lambda texts: model.encode(texts, tok, device=device)


def safe_matmul(a, b):
    """a @ b, silencing numpy's SPURIOUS float32-matmul FPE warnings.

    On some BLAS/SIMD backends numpy's float32 matmul sets the divide/overflow/
    invalid floating-point flags even when every input is finite and bounded —
    numpy then prints `divide by zero / invalid value encountered in matmul`.
    The computed result is correct: our embeddings are always finite and |x| <= 1
    (verified — the warning fires even on the random-init baseline, where the
    prototypes are unit vectors). So we locally ignore those flags rather than
    scrubbing data that was never actually bad.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b


def mine_hard_negatives(encode, train_groups, top_k=1):
    """Offline hard-negative mining, driven by the Encoder callable.

    Embed every training query with the CURRENT encoder, and for each query find
    its top_k most-similar DIFFERENT-intent queries. Those confusable cross-intent
    neighbours are the "hard" negatives. Returns {intent: [hard_neg_text, ...]} —
    the pool run_epoch samples from, one (or k) per anchor each batch.

    Why this helps (beyond the in-batch negatives we already have):
        In one-pair-per-intent batches, each anchor's negative for intent X is just
        whatever ONE X-query happened to be sampled — a RANDOM representative. After
        a few epochs the easy ones give ~0 gradient and quality plateaus. Mining
        instead injects the WORST-CASE confuser (the single most X-looking query for
        this anchor), so the loss keeps sharpening exactly the boundaries still
        broken — e.g. media vs smart_home. Well-separated intents surface only ~0-sim
        negatives, so the signal is spent where it's actually needed.

    Offline = computed from a weight SNAPSHOT; re-mine periodically as the model
    improves and yesterday's hard negs become easy. Same idea as
    customer_intent_search, but here it depends only on `encode` (no model/tok).
    """
    texts, labels = [], []
    for intent, qs in train_groups.items():
        texts.extend(qs)
        labels.extend([intent] * len(qs))
    pool = {it: [] for it in train_groups}
    if not texts:
        return pool
    emb = encode(texts)  # [N, D], L2-normalized
    sim = safe_matmul(emb, emb.T)  # [N, N] cosine
    labels_arr = np.array(labels)
    for i in range(len(texts)):
        scores = sim[i].copy()
        scores[labels_arr == labels[i]] = -2.0  # mask same-intent (incl. self)
        for j in np.argsort(-scores)[:top_k]:
            pool[labels[i]].append(texts[j])
    return pool


def build_prototypes(encode, train_groups):
    """prototype[intent] = L2-normalized MEAN of that intent's train embeddings.
    `encode` is any Encoder callable (texts -> [N, D] L2-normalized).

    Returns (protos, intents):
        protos  [C, D]  one unit vector per intent (the centroids)
        intents [C]     intent names, in train_groups insertion order
    Averaging unit vectors shrinks the length, so we re-normalize (1e-9 guards
    the divide-by-zero if an intent somehow had no examples).
    """
    intents = list(train_groups.keys())
    protos = []
    for it in intents:
        e = encode(train_groups[it])  # [n,D], L2-normed
        m = e.mean(axis=0)  # centroid
        protos.append(m / (np.linalg.norm(m) + 1e-9))  # back to unit length
    return np.stack(protos), intents  # [C,D], [C]


def proto_recall1(encode, train_groups, test_groups) -> float:
    """Centroid-classifier Recall@1 over held-out test queries, for ANY encoder.
    Each test query is labelled by its nearest prototype (cosine). This is the
    honest generalization signal — test.json uses unseen phrasings AND entities.

    No grad / eval-mode handling here: the Encoder owns that (our model.encode runs
    under eval()+no_grad(); run_epoch re-enables train() at the start of each epoch).
    """
    protos, intents = build_prototypes(encode, train_groups)
    correct = total = 0
    for ti, it in enumerate(intents):
        q = encode(test_groups[it])
        if len(q) == 0:
            continue
        pred = safe_matmul(q, protos.T).argmax(axis=1)  # nearest prototype
        correct += int((pred == ti).sum())
        total += len(q)
    return correct / max(1, total)
