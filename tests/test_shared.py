"""The nearest-prototype kernel in shared.py. It depends only on an Encoder
callable (texts -> [N, D] L2-normalized ndarray), so we test it with a tiny fake
encoder instead of the real model — fast and fully deterministic."""

import numpy as np
import shared


def make_fake_encoder(table):
    """Encoder backed by a dict {text: vector}; rows are returned L2-normalized.
    Returns a (0, D) array for empty input, matching the Encoder contract that
    shared.proto_recall1 relies on for intents with no queries."""
    dim = len(next(iter(table.values())))

    def encode(texts):
        if len(texts) == 0:
            return np.empty((0, dim))
        vecs = np.array([table[t] for t in texts], dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    return encode


def test_safe_matmul_matches_plain_matmul():
    a = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    b = np.random.default_rng(1).normal(size=(8, 3)).astype(np.float32)
    assert np.allclose(shared.safe_matmul(a, b), a @ b)


def test_build_prototypes_unit_length_and_intent_order():
    enc = make_fake_encoder(
        {
            "x1": [1.0, 0.0],
            "x2": [1.0, 0.0],  # media → points along +x
            "y1": [0.0, 1.0],
            "y2": [0.0, 1.0],  # weather → points along +y
        }
    )
    protos, intents = shared.build_prototypes(enc, {"media": ["x1", "x2"], "weather": ["y1", "y2"]})
    assert intents == ["media", "weather"]  # insertion order preserved
    # each prototype is a unit vector
    assert np.allclose(np.linalg.norm(protos, axis=1), 1.0)
    # centroid of two identical unit vectors is that unit vector
    assert np.allclose(protos[0], [1.0, 0.0])
    assert np.allclose(protos[1], [0.0, 1.0])


def test_proto_recall1_perfect_when_classes_separable():
    enc = make_fake_encoder(
        {
            "m_train": [1.0, 0.0],
            "m_test": [0.9, 0.1],
            "w_train": [0.0, 1.0],
            "w_test": [0.1, 0.9],
        }
    )
    train = {"media": ["m_train"], "weather": ["w_train"]}
    test = {"media": ["m_test"], "weather": ["w_test"]}
    assert shared.proto_recall1(enc, train, test) == 1.0


def test_proto_recall1_zero_when_test_points_at_wrong_prototype():
    enc = make_fake_encoder(
        {
            "m_train": [1.0, 0.0],
            "m_test": [0.0, 1.0],  # media test sits on weather
            "w_train": [0.0, 1.0],
            "w_test": [1.0, 0.0],  # weather test sits on media
        }
    )
    train = {"media": ["m_train"], "weather": ["w_train"]}
    test = {"media": ["m_test"], "weather": ["w_test"]}
    assert shared.proto_recall1(enc, train, test) == 0.0


def test_proto_recall1_skips_intents_with_no_test_queries():
    enc = make_fake_encoder({"m_train": [1.0, 0.0], "m_test": [1.0, 0.0], "w_train": [0.0, 1.0]})
    train = {"media": ["m_train"], "weather": ["w_train"]}
    test = {"media": ["m_test"], "weather": []}  # weather has no test rows
    assert shared.proto_recall1(enc, train, test) == 1.0


# ── mine_hard_negatives ───────────────────────────────────────────────────────
# Toy geometry from the teaching demo: one media query leans toward smart and one
# smart query leans toward media, so the two are each other's hardest confuser.
_HN_TABLE = {
    "m_pure": [1.0, 0.0, 0.0],
    "m_lean": [0.8, 0.6, 0.0],  # media query that leans onto smart's axis
    "s_pure": [0.0, 1.0, 0.0],
    "s_lean": [0.6, 0.8, 0.0],  # the confuser (cos with m_lean = 0.96)
    "w1": [0.0, 0.0, 1.0],
    "w2": [0.0, 0.1, 1.0],  # weather: well separated from both
}
_HN_GROUPS = {"media": ["m_pure", "m_lean"], "smart": ["s_pure", "s_lean"], "weather": ["w1", "w2"]}


def test_mine_hard_negatives_only_returns_different_intent_texts():
    pool = shared.mine_hard_negatives(make_fake_encoder(_HN_TABLE), _HN_GROUPS, top_k=1)
    for intent, negs in pool.items():
        own = set(_HN_GROUPS[intent])
        assert negs, f"{intent} got no hard negatives"
        assert all(t not in own for t in negs)  # same-intent is masked out


def test_mine_hard_negatives_picks_the_most_confusable_neighbour():
    pool = shared.mine_hard_negatives(make_fake_encoder(_HN_TABLE), _HN_GROUPS, top_k=1)
    # the media↔smart pair are each other's hardest cross-intent neighbour
    assert "s_lean" in pool["media"]
    assert "m_lean" in pool["smart"]


def test_mine_hard_negatives_pool_size_scales_with_top_k():
    table = {"a1": [1.0, 0.0], "a2": [0.9, 0.1], "b1": [0.0, 1.0], "b2": [0.1, 0.9]}
    groups = {"a": ["a1", "a2"], "b": ["b1", "b2"]}
    pool = shared.mine_hard_negatives(make_fake_encoder(table), groups, top_k=2)
    # top_k negatives per query × queries per intent
    assert len(pool["a"]) == 2 * 2
    assert len(pool["b"]) == 2 * 2
