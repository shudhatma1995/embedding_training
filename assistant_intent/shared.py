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

  make_encoder    - wrap our (model, tok, device) as an Encoder callable.
  build_prototypes- encode → mean → L2-normalize, one centroid per intent.
  proto_recall1   - in-training quality signal: nearest-prototype Recall@1 on test.

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
        e = encode(train_groups[it])                             # [n,D], L2-normed
        m = e.mean(axis=0)                                        # centroid
        protos.append(m / (np.linalg.norm(m) + 1e-9))            # back to unit length
    return np.stack(protos), intents                             # [C,D], [C]


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
        pred = safe_matmul(q, protos.T).argmax(axis=1)          # nearest prototype
        correct += int((pred == ti).sum())
        total += len(q)
    return correct / max(1, total)
