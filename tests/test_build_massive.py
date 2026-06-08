"""build_massive's fetch + assemble logic, exercised OFFLINE by monkeypatching the
network calls. Covers mapping application, train-leak + duplicate dropping, per-intent
balancing, and REST depagination — no HTTP and no MASSIVE download."""

import json

import build_massive as bm


def test_build_maps_drops_leaks_and_dupes(monkeypatch, tmp_path):
    # train.json carries one text MASSIVE will also yield -> it must be leak-dropped
    (tmp_path / "train.json").write_text(json.dumps([{"text": "play jazz", "intent": "media"}]))
    monkeypatch.setattr(bm, "DATA_DIR", str(tmp_path))
    fake = [
        ("play_music", "play jazz"),  # media, but == a train text -> leak
        ("play_music", "put on rock"),  # media
        ("iot_wemo_on", "turn on the fan"),  # -> none (unsupported device)
        ("music_query", "whats playing"),  # ambiguous -> dropped (unmapped)
        ("play_music", "put on rock"),  # duplicate -> dropped
        ("weather_query", "will it rain"),  # weather
    ]
    monkeypatch.setattr(bm, "fetch_rows", lambda split: iter(fake))

    rows, meta = bm.build("test", cap_per_intent=10, none_cap=10, seed=0)
    by_text = {r["text"]: r["intent"] for r in rows}

    assert "play jazz" not in by_text  # leak removed
    assert by_text["put on rock"] == "media"
    assert by_text["turn on the fan"] == "none"
    assert by_text["will it rain"] == "weather"
    assert meta["fetched"] == len(fake)
    assert meta["dropped"] == {"unmapped_or_ambiguous": 1, "train_leak": 1, "duplicate": 1}
    assert meta["kept_total"] == len(rows) == 3


def test_build_caps_real_intents_and_none_separately(monkeypatch, tmp_path):
    (tmp_path / "train.json").write_text(json.dumps([]))
    monkeypatch.setattr(bm, "DATA_DIR", str(tmp_path))
    fake = [("play_music", f"song {i}") for i in range(8)]
    fake += [("general_joke", f"joke {i}") for i in range(8)]
    monkeypatch.setattr(bm, "fetch_rows", lambda split: iter(fake))

    _, meta = bm.build("test", cap_per_intent=3, none_cap=5, seed=0)
    assert meta["kept_per_intent"]["media"] == 3  # real intent capped
    assert meta["kept_per_intent"]["none"] == 5  # none capped on its own


def test_fetch_rows_decodes_intent_codes_and_depaginates(monkeypatch):
    names = ["play_music", "weather_query"]
    info = {"dataset_info": {"features": {"intent": {"names": names}}}}
    pages = {
        0: {
            "rows": [{"row": {"intent": 0, "utt": "a"}}, {"row": {"intent": 1, "utt": "b"}}],
            "num_rows_total": 3,
        },
        2: {"rows": [{"row": {"intent": 0, "utt": "c"}}], "num_rows_total": 3},
    }

    def fake_get(url):
        if "/info" in url:
            return info
        offset = int(url.split("offset=")[1].split("&")[0])
        return pages[offset]

    monkeypatch.setattr(bm, "_get_json", fake_get)
    out = list(bm.fetch_rows("test"))
    assert out == [("play_music", "a"), ("weather_query", "b"), ("play_music", "c")]
