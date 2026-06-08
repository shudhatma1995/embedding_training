"""
build_fasttext.py  -  cache pretrained fastText vectors for our vocabulary (lever C).
================================================================================
Produces models/ft_cache/{matrix.npy, vocab.json, meta.json}: a SMALL embedding table
holding a pretrained fastText vector for every word in our train+eval vocabulary.
`encoder.FastTextEncoder` loads this table (frozen) instead of learning word vectors
from scratch — injecting the semantic PRIORS that 1065 rows can't learn (and giving
real vectors to words our 512-word fitted vocab treated as [UNK]).

Why a cache (infra notes, mirrors build_massive.py):
  - The full fastText model is ~1GB; we only need vectors for ~1.3k words. So we load
    it once, look those up, and DISCARD it — training/eval never hold 1GB in RAM.
  - The cache is NOT committed (it is derived third-party data); it is gitignored under
    models/ and regenerated on demand. The committed, reviewed artifacts are this
    builder + the encoder + their tests.
  - No label leakage: we look up *character-derived, label-free* vectors for words that
    happen to appear in eval text — the exact offline stand-in for "vectorize each token
    with fastText at inference (incl. OOV)". No training/fitting touches eval text.

Dependency: gensim (imported lazily, only for the run-once build). Source:
  fastText `wiki-news-300d-1M-subword` via gensim-data (CC-BY-SA), 1M words, 300d.

Run:  python build_fasttext.py        (downloads ~1GB on first run, writes models/ft_cache/)
"""

import json
import os
import re

import numpy as np
from data import DATA_DIR
from intents import NONE_ID, REAL_INTENTS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "models", "ft_cache")
GENSIM_MODEL = "fasttext-wiki-news-subwords-300"
EVAL_FILES = ["train.json", "test.json", "test_wild.json", "test_massive.json"]
SPECIALS = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}


def union_vocab(paths):
    """Sorted, de-duped, lowercased words across the given {text,intent} json files.
    Same `[a-z0-9]+` normalization as the tokenizers, so ids line up with how text is
    tokenized at train/eval time."""
    words = set()
    for p in paths:
        with open(p) as f:
            for row in json.load(f):
                words.update(re.findall(r"[a-z0-9]+", row["text"].lower()))
    return sorted(words)


def assemble(words, get_vector):
    """Build (vocab {word:id}, matrix [V,D], stats) from a `get_vector(word)->vec|None`.

    Words whose vector is None (unavailable) are dropped from the vocab and become [UNK]
    at encode time. Row layout: [PAD]=zeros, [UNK]=mean-of-kept ("average word"),
    [CLS]=zeros placeholder (the encoder grafts a LEARNED [CLS] at position 0), then one
    row per kept word. Pure + deterministic, so it is unit-tested with a fake get_vector.
    """
    kept, vecs = [], []
    for w in words:
        v = get_vector(w)
        if v is not None:
            kept.append(w)
            vecs.append(np.asarray(v, dtype=np.float32))
    if not vecs:
        raise ValueError("no vectors produced — is the fastText model loaded?")
    arr = np.stack(vecs)  # [K, D]
    dim = arr.shape[1]
    vocab = dict(SPECIALS)
    for i, w in enumerate(kept):
        vocab[w] = len(SPECIALS) + i
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    matrix[SPECIALS["[UNK]"]] = arr.mean(axis=0)  # PAD/CLS rows stay zeros
    matrix[len(SPECIALS) :] = arr
    stats = {"total": len(words), "kept": len(kept), "miss": len(words) - len(kept)}
    return vocab, matrix, stats


def load_fasttext_lookup(model_name=GENSIM_MODEL):  # pragma: no cover - network + 1GB model
    """Load fastText via gensim and return (get_vector, oov_supported, meta).
    `get_vector(word)` returns a vector for in-vocab words, an n-gram-composed vector for
    OOV words IF the model supports it, else None (-> [UNK])."""
    import gensim.downloader as api

    kv = api.load(model_name)
    oov_supported = True
    try:
        _ = kv["zzqwxunseennonsenseword"]
    except KeyError:
        oov_supported = False

    def get_vector(word):
        if word in kv.key_to_index:
            return kv[word]
        if oov_supported:
            return kv[word]
        return None

    meta = {
        "model": model_name,
        "known_words": len(kv.key_to_index),
        "oov_supported": oov_supported,
    }
    return get_vector, oov_supported, meta


def main():  # pragma: no cover - run-once build (network + 1GB model)
    words = union_vocab([os.path.join(DATA_DIR, f) for f in EVAL_FILES])
    print(f"union vocab (train+eval, {len(REAL_INTENTS)} real + {NONE_ID}): {len(words)} words")
    print(f"loading {GENSIM_MODEL} (~1GB on first run) ...")
    get_vector, oov_supported, meta = load_fasttext_lookup()
    print(f"  known words: {meta['known_words']}, ngram-OOV supported: {oov_supported}")

    vocab, matrix, stats = assemble(words, get_vector)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "matrix.npy"), matrix)
    with open(os.path.join(OUT_DIR, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    meta.update(stats, dim=int(matrix.shape[1]))
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cov = 100 * stats["kept"] / max(1, stats["total"])
    print(
        f"  coverage: {stats['kept']}/{stats['total']} ({cov:.1f}%); miss->[UNK]: {stats['miss']}"
    )
    print(f"  saved matrix {matrix.shape} + vocab ({len(vocab)}) + meta -> {OUT_DIR}")


if __name__ == "__main__":
    main()
