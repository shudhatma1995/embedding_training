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

import json
import os
import random
from collections import defaultdict

import torch
from intents import NONE_ID  # taxonomy: the out-of-scope label

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")  # canonical home of train.json / test.json


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


# ── eval-set registry ─────────────────────────────────────────────────────────
# The catalog of evaluation sets. Every one is the SAME {text,intent} schema in a
# file under data/, so a single loader serves them all — adding a new external set
# (e.g. SLURP) is a one-line entry here, not a new bespoke loader. `_BUILD_HINT`
# carries the "how to produce this file" message for sets that aren't committed.
EVAL_SETS = {
    "template": "test.json",  # in-distribution, from data_gen/ (committed)
    "wild": "test_wild.json",  # hand-written OOD, from eval_wild.py (committed)
    "massive": "test_massive.json",  # external real data, from build_massive.py (gitignored)
}
_BUILD_HINT = {
    "massive": "run `python build_massive.py` first (needs network)",
    "wild": "run `python eval_wild.py` first",
}


def load_eval_set(name, real_intents, data_dir=DATA_DIR, train_file="train.json"):
    """The evaluation trio for a NAMED eval set (see EVAL_SETS):
        (train_groups, test_groups, none_texts)
    Prototypes are always built from `train_file`; the queries come from the named
    set's file. train/test groups are {intent: [texts]}; none_texts is a flat list.
    One loader for template / wild / massive (and any future set) — no per-set
    duplication, no repeated `[NONE_ID][NONE_ID]` boilerplate."""
    if name not in EVAL_SETS:
        raise ValueError(f"unknown eval set {name!r}; available: {sorted(EVAL_SETS)}")
    test_path = os.path.join(data_dir, EVAL_SETS[name])
    if not os.path.exists(test_path):
        hint = _BUILD_HINT.get(name, "")
        raise FileNotFoundError(f"{test_path} not found" + (f" — {hint}" if hint else ""))
    train_groups = load_by_intent(os.path.join(data_dir, train_file), real_intents)
    test_groups = load_by_intent(test_path, real_intents)
    none_texts = load_by_intent(test_path, [NONE_ID])[NONE_ID]
    return train_groups, test_groups, none_texts


def load_eval_data(real_intents, data_dir=DATA_DIR):
    """Convenience for the common case: the in-distribution `template` eval trio.
    Thin wrapper over load_eval_set so existing callers (evaluate.py, experiments.py,
    the baselines) keep their terse `load_eval_data(REAL_INTENTS)` call."""
    return load_eval_set("template", real_intents, data_dir=data_dir)


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
    Each yield: (anchor_texts, positive_texts, intents) of length == n_intents.
    `intents` is the per-row intent label, so the caller can look up each anchor's
    hard-negative pool (hard-neg mining); plain MNR can just ignore it.
    Incomplete trailing batches are dropped to keep the loss matrix square.
    """
    by = defaultdict(list)
    for idx, intent in enumerate(I):
        by[intent].append(idx)  # group pair-indices by intent
    for v in by.values():
        rng.shuffle(v)
    n_intents = len(by)
    rounds = max(len(v) for v in by.values())
    for r in range(rounds):
        batch = [v[r] for v in by.values() if r < len(v)]
        if len(batch) == n_intents:  # full batch only
            yield [A[i] for i in batch], [P[i] for i in batch], [I[i] for i in batch]


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
