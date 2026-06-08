"""
massive_mapping.py  -  map Amazon MASSIVE's 60 intents onto OUR 8 + none taxonomy.
================================================================================

Why this file exists
  MASSIVE (AmazonScience/massive, CC-BY-4.0) is 19.5k REAL virtual-assistant
  utterances labelled with 60 intents across 18 scenarios. It is third-party data
  we did NOT author — which is exactly why it's the honest external test our own
  synthetic/wild sets can't be (we wrote both of those). To use it we must align
  its label space to ours, and that alignment is a JUDGEMENT we make explicit and
  test here, rather than burying it in a data-loading script.

The mapping is pure (no I/O), so it is fully unit-testable offline. build_massive.py
applies it to the fetched rows; nothing else decides labels.

How each MASSIVE intent is resolved (decisions follow intents.py — the annotation
guideline — NOT ad-hoc taste):
  - Mapped to one of our 8 REAL intents when the scenario is unambiguously one we
    support. Several calls come straight from the guideline:
      * audio_volume_* -> media   (intents.py: "volume" is a media-playback slot)
      * iot_hue_*       -> smart_home, but iot_wemo_*/iot_cleaning/iot_coffee ->
        none, because smart_home is scoped to LIGHTS in v1 (guideline) — other
        devices are unsupported.
      * datetime_*      -> answers ("what time is it in tokyo -> answers")
      * transport_query/traffic -> navigation, but transport_taxi/ticket -> none
        ("booking a ride/ticket -> none", "call me an uber -> none").
  - Mapped to `none` when the scenario is a REAL capability we deliberately don't
    support (recommendations, takeaway, cooking recipes, news, non-light devices,
    ride/ticket booking, chit-chat). These are the most valuable none examples:
    genuine OOS, not gibberish.
  - DROPPED (excluded from the eval) when alignment is genuinely ambiguous w.r.t.
    our taxonomy — e.g. music_query ("what's playing" straddles media vs answers),
    recommendation_locations (overlaps navigation's "nearby places"), contact
    CRUD, social posting. We drop rather than guess so a mislabel of OURS can't be
    scored as a model error. Silent guessing would corrupt the very yardstick.

Snapshot note: MASSIVE_INTENTS pins the 60 labels we mapped against. If upstream
adds/renames an intent, the completeness test fails loudly instead of silently
dropping the new intent to `none`.
"""

from intents import NONE_ID, REAL_INTENTS

# The 60 MASSIVE intents (en-US, dataset rev as fetched). Pinned so the mapping's
# completeness can be asserted against a known set (see tests).
MASSIVE_INTENTS = frozenset(
    {
        "alarm_query", "alarm_remove", "alarm_set",
        "audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up",
        "calendar_query", "calendar_remove", "calendar_set",
        "cooking_query", "cooking_recipe",
        "datetime_convert", "datetime_query",
        "email_addcontact", "email_query", "email_querycontact", "email_sendemail",
        "general_greet", "general_joke", "general_quirky",
        "iot_cleaning", "iot_coffee",
        "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff",
        "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on",
        "lists_createoradd", "lists_query", "lists_remove",
        "music_dislikeness", "music_likeness", "music_query", "music_settings",
        "news_query",
        "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio",
        "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock",
        "recommendation_events", "recommendation_locations", "recommendation_movies",
        "social_post", "social_query",
        "takeaway_order", "takeaway_query",
        "transport_query", "transport_taxi", "transport_ticket", "transport_traffic",
        "weather_query",
    }
)  # fmt: skip

# MASSIVE intents grouped by the OUR-intent they map to. Grouping (vs one flat dict)
# keeps the decisions readable and reviewable side by side.
_REAL_GROUPS = {
    "media": [
        "play_music", "play_radio", "play_audiobook", "play_podcasts",
        "music_likeness", "music_dislikeness", "music_settings",
        "audio_volume_up", "audio_volume_down", "audio_volume_mute", "audio_volume_other",
    ],
    "smart_home": [
        "iot_hue_lighton", "iot_hue_lightoff", "iot_hue_lightdim",
        "iot_hue_lightup", "iot_hue_lightchange",
    ],
    "productivity": [
        "calendar_set", "calendar_query", "calendar_remove",
        "lists_createoradd", "lists_remove", "lists_query",
    ],
    "communication": ["email_sendemail", "email_query"],
    "timers_alarms": ["alarm_set", "alarm_query", "alarm_remove"],
    "weather": ["weather_query"],
    "navigation": ["transport_query", "transport_traffic"],
    "answers": [
        "qa_factoid", "qa_definition", "qa_maths", "qa_currency", "qa_stock",
        "datetime_query", "datetime_convert",
    ],
}  # fmt: skip

# Real capabilities we deliberately don't support -> genuine out-of-scope (`none`).
_NONE_INTENTS = [
    "general_greet", "general_joke", "general_quirky",  # chit-chat
    "recommendation_events", "recommendation_movies",  # recommendations
    "cooking_recipe",  # recipe help
    "takeaway_query", "takeaway_order",  # food ordering
    "iot_wemo_on", "iot_wemo_off", "iot_cleaning", "iot_coffee",  # non-light devices
    "transport_taxi", "transport_ticket",  # ride/ticket booking
    "news_query",  # news
]  # fmt: skip

# Ambiguous w.r.t. our taxonomy -> excluded so our own mislabels can't be scored as
# the model's errors. (music_query: media vs answers; recommendation_locations:
# overlaps navigation; *contact: contact CRUD, not reaching a person; social_*:
# social media; cooking_query: overlaps answers; play_game: not playback nor clearly none.)
MASSIVE_DROP = frozenset(
    {
        "play_game", "music_query",
        "email_addcontact", "email_querycontact",
        "social_query", "social_post",
        "recommendation_locations", "cooking_query",
    }
)  # fmt: skip

# Flatten the groups into the lookup the builder uses: {massive_intent: our_label}.
MASSIVE_TO_INTENT = {m: NONE_ID for m in _NONE_INTENTS}
for _label, _members in _REAL_GROUPS.items():
    for _m in _members:
        MASSIVE_TO_INTENT[_m] = _label


def map_intent(massive_intent: str):
    """Our label for a MASSIVE intent, or None if it is dropped / unknown.

    Returns one of REAL_INTENTS, NONE_ID, or None. None means "exclude this row":
    either a deliberately-dropped ambiguous intent or an intent not in our pinned
    snapshot (e.g. an upstream addition we haven't reviewed)."""
    return MASSIVE_TO_INTENT.get(massive_intent)


def target_intents():
    """The OUR-intent labels MASSIVE can produce: REAL_INTENTS plus `none`."""
    return [*REAL_INTENTS, NONE_ID]
