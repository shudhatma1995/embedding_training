"""I/O and batch-shaping in data.py. The headline test is the batching invariant:
each training batch must contain EXACTLY ONE pair per intent, so every off-diagonal
entry in the contrastive loss is a genuine negative (no false negatives)."""

import json
import random

import data


def _write_rows(tmp_path, rows):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rows))
    return str(path)


def test_load_by_intent_filters_and_follows_keep_order(tmp_path):
    rows = [
        {"text": "play jazz", "intent": "media"},
        {"text": "lights off", "intent": "smart_home"},
        {"text": "skip song", "intent": "media"},
        {"text": "call mom", "intent": "communication"},  # not in keep
    ]
    path = _write_rows(tmp_path, rows)
    groups = data.load_by_intent(path, ["smart_home", "media"])

    assert list(groups.keys()) == ["smart_home", "media"]  # keep order, not data order
    assert groups["media"] == ["play jazz", "skip song"]
    assert groups["smart_home"] == ["lights off"]


def test_load_by_intent_kept_intent_with_no_rows_is_empty(tmp_path):
    path = _write_rows(tmp_path, [{"text": "play jazz", "intent": "media"}])
    groups = data.load_by_intent(path, ["media", "weather"])
    assert groups["weather"] == []


def test_intents_in_file_is_sorted_and_unique(tmp_path):
    rows = [
        {"text": "a", "intent": "media"},
        {"text": "b", "intent": "answers"},
        {"text": "c", "intent": "media"},
    ]
    path = _write_rows(tmp_path, rows)
    assert data.intents_in_file(path) == ["answers", "media"]


def test_make_pairs_are_same_intent_and_two_distinct_utterances():
    groups = {"media": ["m0", "m1", "m2", "m3"], "weather": ["w0", "w1"]}
    a, p, i = data.make_pairs(groups, pairs_per_intent=2, rng=random.Random(0))
    assert len(a) == len(p) == len(i)
    for anchor, positive, intent in zip(a, p, i):
        assert anchor != positive  # a positive pair is two DIFFERENT utterances
        assert anchor.startswith(intent[0]) and positive.startswith(intent[0])


def test_make_pairs_capacity_is_min_of_request_and_half_the_pool():
    groups = {"media": ["m0", "m1", "m2", "m3", "m4"]}  # 5 → at most 2 pairs
    _, _, i = data.make_pairs(groups, pairs_per_intent=10, rng=random.Random(1))
    assert i.count("media") == 2


def test_make_pairs_is_deterministic_under_same_seed():
    groups = {"a": ["a0", "a1", "a2", "a3"], "b": ["b0", "b1", "b2", "b3"]}
    out1 = data.make_pairs(groups, 2, random.Random(123))
    out2 = data.make_pairs(groups, 2, random.Random(123))
    assert out1 == out2


def test_iter_batches_has_exactly_one_pair_per_intent():
    # anchors carry their intent in the first character so we can check the invariant.
    a = ["a0", "a1", "b0", "b1", "c0"]
    p = ["a0p", "a1p", "b0p", "b1p", "c0p"]
    i = ["a", "a", "b", "b", "c"]
    n_intents = len(set(i))

    batches = list(data.iter_batches(a, p, i, rng=random.Random(0)))
    assert batches, "expected at least one full batch"
    for anchors, _positives in batches:
        assert len(anchors) == n_intents
        # every batch covers each intent exactly once → no false negatives
        intents_in_batch = [t[0] for t in anchors]
        assert sorted(intents_in_batch) == sorted(set(i))
        assert len(set(intents_in_batch)) == len(intents_in_batch)


def test_iter_batches_drops_incomplete_trailing_batch():
    # intent "c" has only one pair → only one full (3-intent) batch is possible.
    a = ["a0", "a1", "b0", "b1", "c0"]
    p = ["x"] * 5
    i = ["a", "a", "b", "b", "c"]
    assert len(list(data.iter_batches(a, p, i, random.Random(0)))) == 1
