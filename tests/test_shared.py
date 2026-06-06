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
