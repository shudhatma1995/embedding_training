"""The MASSIVE -> our-taxonomy alignment in massive_mapping.py. Pure data, so these
run offline with no network and no model — they guard the JUDGEMENT, not the I/O."""

import massive_mapping as mm
from intents import NONE_ID, REAL_INTENTS


def test_every_massive_intent_is_either_mapped_or_dropped():
    # the keystone invariant: no MASSIVE intent is silently forgotten. Mapped and
    # dropped together must cover the pinned 60 exactly — no extras, no omissions.
    covered = set(mm.MASSIVE_TO_INTENT) | set(mm.MASSIVE_DROP)
    assert covered == set(mm.MASSIVE_INTENTS)


def test_mapped_and_dropped_are_disjoint():
    assert set(mm.MASSIVE_TO_INTENT).isdisjoint(mm.MASSIVE_DROP)


def test_pinned_snapshot_has_60_intents():
    assert len(mm.MASSIVE_INTENTS) == 60


def test_every_mapped_value_is_a_valid_target_label():
    valid = {*REAL_INTENTS, NONE_ID}
    assert set(mm.MASSIVE_TO_INTENT.values()) <= valid


def test_map_intent_returns_none_for_dropped_and_unknown():
    assert mm.map_intent("music_query") is None  # dropped
    assert mm.map_intent("not_a_real_massive_intent") is None  # unknown


def test_guideline_driven_decisions():
    # the non-obvious calls that follow intents.py's boundary rules — pin them so a
    # careless edit to the mapping trips a test instead of skewing the eval.
    assert mm.map_intent("audio_volume_up") == "media"  # volume is a media slot
    assert mm.map_intent("iot_hue_lighton") == "smart_home"
    assert mm.map_intent("iot_wemo_on") == NONE_ID  # non-light device -> unsupported
    assert mm.map_intent("iot_coffee") == NONE_ID
    assert mm.map_intent("datetime_query") == "answers"  # "what time is it" -> answers
    assert mm.map_intent("transport_query") == "navigation"
    assert mm.map_intent("transport_taxi") == NONE_ID  # booking a ride -> none
    assert mm.map_intent("transport_ticket") == NONE_ID
    assert mm.map_intent("recommendation_movies") == NONE_ID
    assert mm.map_intent("qa_factoid") == "answers"
    assert mm.map_intent("alarm_set") == "timers_alarms"


def test_real_intents_are_all_reachable_from_massive():
    # every one of our 8 real intents has at least one MASSIVE source intent, so the
    # external eval actually exercises the whole taxonomy (not a lopsided subset).
    produced = set(mm.MASSIVE_TO_INTENT.values()) - {NONE_ID}
    assert produced == set(REAL_INTENTS)


def test_none_bucket_is_populated():
    none_sources = [m for m, lab in mm.MASSIVE_TO_INTENT.items() if lab == NONE_ID]
    assert len(none_sources) >= 10  # plenty of genuine out-of-scope sources
