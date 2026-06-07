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
    for anchors, _positives, intents in batches:
        assert len(anchors) == n_intents
        # every batch covers each intent exactly once → no false negatives
        intents_in_batch = [t[0] for t in anchors]
        assert sorted(intents_in_batch) == sorted(set(i))
        assert len(set(intents_in_batch)) == len(intents_in_batch)
        # the yielded per-row intents must match each anchor's own intent (run_epoch
        # uses them to look up the right hard-negative pool)
        assert intents == intents_in_batch


def test_iter_batches_drops_incomplete_trailing_batch():
    # intent "c" has only one pair → only one full (3-intent) batch is possible.
    a = ["a0", "a1", "b0", "b1", "c0"]
    p = ["x"] * 5
    i = ["a", "a", "b", "b", "c"]
    assert len(list(data.iter_batches(a, p, i, random.Random(0)))) == 1


# ── eval-set registry ─────────────────────────────────────────────────────────
def _fake_data_dir(tmp_path):
    """A self-contained data dir: train.json + a 'template' test.json (test.json is
    the EVAL_SETS filename for 'template'), so load_eval_set runs fully offline."""
    (tmp_path / "train.json").write_text(
        json.dumps(
            [{"text": "play jazz", "intent": "media"}, {"text": "nonsense", "intent": "none"}]
        )
    )
    (tmp_path / "test.json").write_text(
        json.dumps(
            [{"text": "put on music", "intent": "media"}, {"text": "blah", "intent": "none"}]
        )
    )
    return str(tmp_path)


def test_eval_sets_registry_has_the_known_sets():
    assert {"template", "wild", "massive"} <= set(data.EVAL_SETS)


def test_load_eval_set_returns_train_test_none_trio(tmp_path):
    d = _fake_data_dir(tmp_path)
    train, test, none = data.load_eval_set("template", ["media"], data_dir=d)
    assert train["media"] == ["play jazz"]
    assert test["media"] == ["put on music"]
    assert none == ["blah"]  # none pulled from the SAME test file


def test_load_eval_data_is_the_template_set(tmp_path):
    # the convenience wrapper must be byte-identical to the registry call it delegates to
    d = _fake_data_dir(tmp_path)
    assert data.load_eval_data(["media"], data_dir=d) == data.load_eval_set(
        "template", ["media"], data_dir=d
    )


def test_load_eval_set_rejects_unknown_name(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="unknown eval set"):
        data.load_eval_set("nope", ["media"], data_dir=str(tmp_path))


def test_load_eval_set_missing_file_raises_with_hint(tmp_path):
    import pytest

    (tmp_path / "train.json").write_text(json.dumps([{"text": "x", "intent": "media"}]))
    # 'massive' file is absent → FileNotFoundError carrying the build hint
    with pytest.raises(FileNotFoundError, match="build_massive"):
        data.load_eval_set("massive", ["media"], data_dir=str(tmp_path))
