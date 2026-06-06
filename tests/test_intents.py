"""The taxonomy in intents.py is the source of truth for labels — these tests
guard its internal consistency so data and model can never drift from the spec."""

import intents


def test_none_id_is_none():
    assert intents.NONE_ID == "none"
    assert intents.NONE_ID in intents.INTENT_IDS


def test_real_intents_excludes_none_and_keeps_order():
    assert intents.NONE_ID not in intents.REAL_INTENTS
    # REAL_INTENTS is INTENT_IDS minus none, order preserved.
    assert intents.REAL_INTENTS == [i for i in intents.INTENT_IDS if i != "none"]


def test_intent_ids_are_unique():
    assert len(intents.INTENT_IDS) == len(set(intents.INTENT_IDS))


def test_spec_keys_match_intent_ids_exactly():
    # No orphan specs, no intent missing a spec.
    assert set(intents.INTENT_SPEC) == set(intents.INTENT_IDS)


def test_every_spec_has_required_nonempty_fields():
    for iid, spec in intents.INTENT_SPEC.items():
        assert spec["definition"].strip(), f"{iid} has an empty definition"
        assert spec["examples"], f"{iid} has no examples"
        assert spec["not"], f"{iid} has no boundary ('not') rules"
