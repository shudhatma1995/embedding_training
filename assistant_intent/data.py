"""
data.py  -  all the non-training I/O and batch-shaping for the intent pipeline.
================================================================================
"Side" code only — reading the json, turning queries into positive pairs, shaping
those pairs into one-pair-per-intent batches, and writing the trained artifacts.
train.py imports from here so it can stay focused on the learning algorithm itself
(loss, forward, epoch loop). evaluate.py / experiments.py reuse `load_eval_data`.

Kept dependency-light (json + os + random + torch) and NOT named `evaluate`, so
nothing on sys.path can shadow it (train.py puts customer_intent_search/ — which
has its own sentence_transformers-importing evaluate.py — at the front of sys.path).
"""
import os
import json
import random
from collections import defaultdict

import torch

from intents import NONE_ID   # taxonomy: the out-of-scope label

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")   # canonical home of train.json / test.json


# ── reading ──────────────────────────────────────────────────────────────────
def load_by_intent(path: str, keep: list) -> dict:
    """Load [{text,intent}] json → {intent: [texts]} for the kept intents.
    Keys follow `keep` order; an intent with no rows maps to an empty list."""
    with open(path) as f:
        rows = json.load(f)
    groups = {k: [] for k in keep}
    for r in rows:
        if r["intent"] in keep:
            groups[r["intent"]].append(r["text"])
    return groups


def intents_in_file(path: str) -> list:
    """Return the sorted unique intent labels actually present in a {text,intent} json.
    Used to validate a requested --intents subset against what the data really has."""
    with open(path) as f:
        rows = json.load(f)
    return sorted({r["intent"] for r in rows})


def load_eval_data(real_intents, data_dir=DATA_DIR):
    """The standard evaluation trio for a set of real intents:
        (train_groups, test_groups, none_texts)
    train/test groups are {intent: [texts]}; none_texts is a flat list from test.json.
    One call so evaluate.py, experiments.py, and the pretrained-baseline script all
    load identically (no repeated `[NONE_ID][NONE_ID]` boilerplate)."""
    train_groups = load_by_intent(os.path.join(data_dir, "train.json"), real_intents)
    test_groups = load_by_intent(os.path.join(data_dir, "test.json"), real_intents)
    none_texts = load_by_intent(os.path.join(data_dir, "test.json"), [NONE_ID])[NONE_ID]
    return train_groups, test_groups, none_texts


# ── pairing ──────────────────────────────────────────────────────────────────
def make_pairs(groups: dict, pairs_per_intent: int, rng: random.Random):
    """
    One positive pair = two DIFFERENT utterances of the same intent.
    Re-sampled every epoch (dynamic pairing) so a query meets many partners.
    Returns parallel lists: anchors, positives, intents.
    """
    A, P, I = [], [], []
    for intent, qs in groups.items():
        s = qs[:]
        rng.shuffle(s)
        n = min(pairs_per_intent, len(s) // 2)
        for i in range(0, n * 2, 2):
            A.append(s[i])
            P.append(s[i + 1])
            I.append(intent)
    return A, P, I


def iter_batches(A, P, I, rng: random.Random):
    """
    Yield batches with EXACTLY ONE pair per intent (no false negatives).
    Each yield: (anchor_texts, positive_texts) of length == n_intents.
    Incomplete trailing batches are dropped to keep the loss matrix square.
    """
    by = defaultdict(list)
    for idx, intent in enumerate(I):
        by[intent].append(idx)                 # group pair-indices by intent
    for v in by.values():
        rng.shuffle(v)
    n_intents = len(by)
    rounds = max(len(v) for v in by.values())
    for r in range(rounds):
        batch = [v[r] for v in by.values() if r < len(v)]
        if len(batch) == n_intents:           # full batch only
            yield [A[i] for i in batch], [P[i] for i in batch]


# ── writing ──────────────────────────────────────────────────────────────────
def save_artifacts(output_dir, model, tok, config):
    """Write model.pt, tokenizer.json, config.json to output_dir; return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "model": os.path.join(output_dir, "model.pt"),
        "tokenizer": os.path.join(output_dir, "tokenizer.json"),
        "config": os.path.join(output_dir, "config.json"),
    }
    torch.save(model.state_dict(), paths["model"])
    tok.save(paths["tokenizer"])
    with open(paths["config"], "w") as f:
        json.dump(config, f, indent=2)
    return paths
