"""The `none`-threshold math and the honest val/test split in evaluate.py.
These are pure-numpy functions, so we feed hand-built arrays with known answers."""

import evaluate
import numpy as np


def test_metrics_at_counts_real_and_none_correctly():
    # two real queries (both classified at the correct prototype) and two none.
    real_max = np.array([0.9, 0.4])
    real_correct = np.array([True, True])
    none_max = np.array([0.2, 0.8])
    m = evaluate.metrics_at(0.5, real_max, real_correct, none_max)

    # real is right iff correct AND top-sim >= tau → only the 0.9 query clears 0.5
    assert m["real_acc"] == 0.5
    # none is right iff top-sim < tau → only the 0.2 query is rejected
    assert m["none_recall"] == 0.5
    assert m["overall"] == 0.5  # (1 + 1) / (2 + 2)
    assert m["tau"] == 0.5


def test_metrics_at_threshold_is_inclusive_on_real_side():
    # a real query exactly at tau should clear it (>= tau).
    m = evaluate.metrics_at(0.5, np.array([0.5]), np.array([True]), np.array([]))
    assert m["real_acc"] == 1.0


def test_metrics_at_correct_label_below_tau_is_rejected():
    # right prototype but sim below tau → answered none → not counted as real-correct.
    m = evaluate.metrics_at(0.6, np.array([0.5]), np.array([True]), np.array([]))
    assert m["real_acc"] == 0.0


def test_pick_best_maximizes_overall():
    rows = [
        {"tau": 0.3, "overall": 0.4},
        {"tau": 0.5, "overall": 0.7},
        {"tau": 0.7, "overall": 0.6},
    ]
    assert evaluate.pick_best(rows)["tau"] == 0.5


def test_tau_sweep_covers_all_taus():
    rows = evaluate.tau_sweep(
        np.array([0.8]), np.array([True]), np.array([0.2]), taus=[0.3, 0.5, 0.7]
    )
    assert [r["tau"] for r in rows] == [0.3, 0.5, 0.7]


def test_split_val_test_partitions_without_overlap_and_keeps_classes():
    groups = {"media": [f"m{i}" for i in range(6)], "weather": [f"w{i}" for i in range(4)]}
    none_texts = [f"n{i}" for i in range(8)]
    val_g, val_none, test_g, test_none = evaluate.split_val_test(groups, none_texts, frac=0.5)

    for it, originals in groups.items():
        union = set(val_g[it]) | set(test_g[it])
        assert union == set(originals)  # nothing lost
        assert not (set(val_g[it]) & set(test_g[it]))  # nothing duplicated
        assert val_g[it] and test_g[it]  # both halves keep the class
    assert set(val_none) | set(test_none) == set(none_texts)
    assert not (set(val_none) & set(test_none))


def test_split_val_test_is_deterministic_for_a_seed():
    groups = {"a": [f"a{i}" for i in range(6)]}
    none = [f"n{i}" for i in range(6)]
    s1 = evaluate.split_val_test(groups, none, frac=0.5, seed=7)
    s2 = evaluate.split_val_test(groups, none, frac=0.5, seed=7)
    assert s1 == s2
