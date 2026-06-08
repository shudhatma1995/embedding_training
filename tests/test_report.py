"""report.py pure logic: regression + weak-intent detection and HTML rendering.
No training/IO — fed hand-built run dicts matching bench.py's schema."""

from report import _previous_sibling, regressions, render_html, weak_intents


def _run(ts, massive_r1=0.85, weather_r1=0.60):
    return {
        "name": "exp",
        "timestamp": ts,
        "seeds": [0, 1],
        "eval_sets": ["in-dist", "MASSIVE"],
        "intents": ["media", "weather"],
        "arms": {
            "fasttext": {
                "label": "fastText-frozen",
                "params": 1000,
                "sets": {
                    "in-dist": {
                        "recall1_mean": 0.99,
                        "recall1_std": 0.01,
                        "honest_overall_mean": 0.95,
                        "none_recall_mean": 0.8,
                        "real_sim_mean": 0.9,
                        "none_sim_mean": 0.3,
                        "per_intent": {
                            "media": {"recall1_mean": 0.95, "n": 10},
                            "weather": {"recall1_mean": weather_r1, "n": 10},
                        },
                        "misclassified": [
                            {"text": "<b>hi</b>", "true": "weather", "pred": "media", "sim": 0.7}
                        ],
                    },
                    "MASSIVE": {
                        "recall1_mean": massive_r1,
                        "recall1_std": 0.01,
                        "honest_overall_mean": 0.8,
                        "none_recall_mean": 0.7,
                        "real_sim_mean": 0.8,
                        "none_sim_mean": 0.4,
                        "per_intent": {
                            "media": {"recall1_mean": 0.9, "n": 10},
                            "weather": {"recall1_mean": 0.85, "n": 10},
                        },
                        "misclassified": [],
                    },
                },
            }
        },
    }


def test_regressions_detect_a_real_drop():
    cur, prev = _run("t2", massive_r1=0.80), _run("t1", massive_r1=0.85)
    regs = regressions(cur, prev)
    assert ("fasttext", "MASSIVE") in [(a, s) for a, s, *_ in regs]


def test_no_regression_for_small_change_or_no_previous():
    assert (
        regressions(_run("t2", massive_r1=0.845), _run("t1", massive_r1=0.85)) == []
    )  # −0.5pp < 2pp
    assert regressions(_run("t1"), None) == []  # first run


def test_weak_intents_flags_below_threshold():
    weak = weak_intents(_run("t1", weather_r1=0.60))
    assert ("fasttext", "in-dist", "weather") in [(a, s, it) for a, s, it, _ in weak]
    # and a strong intent is NOT flagged
    assert "media" not in [it for _, _, it, _ in weak]


def test_render_html_escapes_and_includes_sections():
    html_str = render_html(_run("t2", massive_r1=0.80), _run("t1", massive_r1=0.85))
    assert isinstance(html_str, str) and "fastText-frozen" in html_str
    assert "MASSIVE" in html_str and "regression" in html_str.lower()
    assert "&lt;b&gt;hi&lt;/b&gt;" in html_str  # misclassified text is HTML-escaped


def test_render_html_first_run_has_baseline_note():
    assert "First run" in render_html(_run("t1"), None)


def test_previous_sibling_picks_the_run_before(tmp_path):
    for ts in ["20260101T000000_0000", "20260102T000000_0000", "20260103T000000_0000"]:
        (tmp_path / f"{ts}.json").write_text("{}")
    mid = str(tmp_path / "20260102T000000_0000.json")
    assert _previous_sibling(mid).endswith("20260101T000000_0000.json")  # the one before
    first = str(tmp_path / "20260101T000000_0000.json")
    assert _previous_sibling(first) is None  # nothing before the earliest
