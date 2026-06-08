"""bench.py persistence + registry (the pure parts; the multi-seed run is integration)."""

from bench import ARMS, load_previous, save_results


def _run(ts):
    return {
        "name": "exp",
        "timestamp": ts,
        "seeds": [0],
        "eval_sets": [],
        "intents": [],
        "arms": {},
    }


def test_arms_registry_has_expected_builders():
    assert {"word", "random", "fasttext"} <= set(ARMS)


def test_load_previous_none_when_empty(tmp_path):
    assert load_previous("nope", root=str(tmp_path)) is None


def test_save_then_load_previous_returns_latest(tmp_path):
    save_results(_run("2026-06-07T10:00:00+00:00"), root=str(tmp_path))
    save_results(_run("2026-06-07T12:00:00+00:00"), root=str(tmp_path))
    prev = load_previous("exp", root=str(tmp_path))
    assert prev["timestamp"] == "2026-06-07T12:00:00+00:00"  # most recent
