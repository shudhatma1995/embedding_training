"""
intents.py  -  SOURCE OF TRUTH for the assistant intent classifier's labels.
============================================================================

This file is the *annotation guideline*: the rules every training and test
example is checked against. When you (or an LLM helper) write a new example,
you label it by reading the rules here -- that is what keeps thousands of
examples mutually consistent instead of silently contradicting each other.

THE LABELING PRINCIPLE
----------------------
An intent = a distinct downstream ACTION (a different handler / API / response
type). If a distinction is just a parameter the handler can read off the
utterance, it is a SLOT, not a new intent:
    - lights on vs off            -> one intent (smart_home), on/off is a slot
    - media play vs pause vs skip -> one intent (media),      verb is a slot
Labels should sit at a CONSISTENT level of granularity.

V1 schema = 5 flat intents (no hierarchy yet; we earn hierarchy only when a
theme grows to ~4+ distinct actions or coarse classes start getting confused).
"""

# Stable ids. Data files and the trained model refer to these strings —
# don't rename casually once data exists.
INTENT_IDS = ["answers", "media", "smart_home", "productivity", "communication",
              "timers_alarms", "weather", "navigation", "none"]

# `none` is the out-of-scope class: handled by a similarity threshold / negatives
# at train time, NEVER trained as a positive prototype (see NONE_HANDLING below).
# REAL_INTENTS is therefore the trainable superset — train.py/evaluate.py select a
# (possibly partial) subset of it via --intents; `none` is excluded by construction.
NONE_ID = "none"
REAL_INTENTS = [i for i in INTENT_IDS if i != NONE_ID]

# Full spec for each intent. `examples` are CANONICAL illustrations for the
# guideline (not the training set — seed phrasings come next, in make_data.py).
INTENT_SPEC = {
    "answers": {
        "definition": (
            "A coherent INFORMATIONAL question the assistant answers from "
            "knowledge. No app or device is involved."
        ),
        "examples": [
            "how tall is mount everest",
            "what's 15 percent of 80",
            "who won the 2022 world cup",
        ],
        "not": [
            "needs the calendar or email -> productivity (even if phrased as a question)",
            "controls a device -> media / smart_home",
            "not a real question / unanswerable -> none",
        ],
    },
    "media": {
        "definition": (
            "Control media PLAYBACK (music, podcasts, video). The verb "
            "(play / pause / skip / volume) is a slot, not a separate intent."
        ),
        "examples": [
            "play some jazz",
            "pause the music",
            "skip to the next track",
            "turn it up",
        ],
        "not": [
            "'who sings this song' -> answers (it's an info question)",
            "'turn on some music' is media; 'turn on the lights' is smart_home",
        ],
    },
    "smart_home": {
        "definition": (
            "Control the lights (on / off / dim / brightness). On vs off and "
            "the room are slots. (Scoped to lights for v1.)"
        ),
        "examples": [
            "turn on the living room lights",
            "lights off",
            "dim the bedroom lights to 50 percent",
        ],
        "not": [
            "'turn on some music' -> media (same verb, different object)",
            "'remind me to turn off the lights' -> productivity (it's a reminder)",
        ],
    },
    "productivity": {
        "definition": (
            "Manage the CALENDAR and REMINDERS/TASKS -- create, read, move, or "
            "cancel events; set reminders -- even when phrased as a question. "
            "(Email/messaging moved out to `communication`.)"
        ),
        "examples": [
            "schedule a meeting with john tomorrow at 3",
            "what's on my calendar today",
            "remind me to call the bank",
            "move my 2pm meeting to 4",
        ],
        "not": [
            "reaching a person (call/text/email) -> communication",
            "'remind me to email mom' -> productivity (it's a reminder, not the act of emailing)",
            "'what time is it in tokyo' -> answers (needs no app)",
            "controlling media/lights -> media / smart_home",
        ],
    },
    "communication": {
        "definition": (
            "Reach a PERSON: place a phone call, send a text/SMS, or compose / "
            "read / reply to EMAIL. The channel (call/text/email) and the contact "
            "are slots."
        ),
        "examples": [
            "call mom",
            "text sarah that i'm on my way",
            "email my boss about the report",
            "read me my latest email",
        ],
        "not": [
            "scheduling time with someone -> productivity ('schedule a call with john')",
            "'call me an uber' -> none (not a person)",
            "'remind me to text dad' -> productivity (it's a reminder)",
        ],
    },
    "timers_alarms": {
        "definition": (
            "Set, cancel, or check a countdown TIMER or a clock ALARM. The "
            "duration or wake-time (and an optional label) are slots. No task "
            "content and no calendar entry."
        ),
        "examples": [
            "set a timer for 10 minutes",
            "set an alarm for 7am",
            "wake me up at 6:30",
            "cancel my timer",
        ],
        "not": [
            "has a task/event -> productivity ('remind me to call the bank')",
            "'what time is it' -> answers",
        ],
    },
    "weather": {
        "definition": (
            "Current conditions or the FORECAST (temperature, rain, snow, wind) "
            "for a place/time. Hits a weather service."
        ),
        "examples": [
            "what's the weather today",
            "will it rain tomorrow",
            "how hot is it going to be",
            "do i need an umbrella",
        ],
        "not": [
            "definitional/scientific knowledge -> answers "
            "('difference between weather and climate')",
        ],
    },
    "navigation": {
        "definition": (
            "Directions, ETA, traffic, or finding nearby places -- anything that "
            "needs your LOCATION or live maps. Not booking a ride or a ticket."
        ),
        "examples": [
            "navigate home",
            "how long to the airport",
            "directions to the train station",
            "is there traffic to work",
        ],
        "not": [
            "pure geographic knowledge -> answers ('how far is the moon')",
            "booking a ride/ticket -> none ('call me an uber', 'book a flight')",
        ],
    },
    "none": {
        "definition": (
            "NOT a serveable request: gibberish, background chatter, plain "
            "statements, or capabilities the assistant doesn't support."
        ),
        "examples": [
            "asdfgh",
            "i'm so tired today",
            "book me a flight to paris",
        ],
        "not": [
            "a real info question -> answers",
        ],
    },
}

# Cross-intent tie-breakers — the edge cases that decide ambiguous examples.
# These are the rules to apply when an utterance could plausibly fit two labels.
BOUNDARY_RULES = [
    "App-or-device test: if it needs a specific app/service or device, label it "
    "that intent EVEN IF phrased as a question. 'what time is my meeting' -> "
    "productivity; 'what time is it in tokyo' -> answers.",
    "answers vs none: would the assistant respond with an answer (-> answers) or "
    "decline/ignore (-> none)? 'what flights are cheapest' -> answers; "
    "'book me a flight' -> none (unsupported action).",
    "Shared verbs decide by OBJECT, not verb: 'turn on' + lights -> smart_home; "
    "'turn on' + music -> media. Generate many such near-collisions on purpose.",
    "timers_alarms vs productivity: a bare countdown or wake-time with NO task -> "
    "timers_alarms ('set an alarm for 7am'); a task/event with content -> "
    "productivity ('remind me to call the bank').",
    "communication vs productivity: reaching a person (call/text/email) -> "
    "communication; scheduling time or setting a reminder -> productivity. "
    "'email mom i'll be late' -> communication; 'remind me to email mom' -> productivity.",
    "weather vs answers: actual conditions/forecast -> weather ('will it rain "
    "tomorrow'); definitional knowledge -> answers ('weather vs climate').",
    "navigation vs answers vs none: needs your location/live traffic -> navigation "
    "('how long to the airport'); geographic knowledge -> answers ('how far is the "
    "moon'); booking a ride/ticket -> none ('call me an uber').",
]

# Single-label classifier can't represent two intents at once.
# v1 policy: EXCLUDE compound commands from training data; handle them
# separately downstream. Don't let 'turn off the lights and play music'
# pollute a single-intent label.
MULTI_INTENT_POLICY = "exclude_from_training"

# 'none' is not a tight semantic cluster (it's "everything else"), so it is NOT
# trained as a normal prototype. Mechanism decided at training time — most likely
# a SIMILARITY THRESHOLD (if the nearest real-intent prototype is too far, answer
# 'none') and/or using none examples as negatives. Flagged here so the data step
# still collects none examples for evaluating that threshold.
NONE_HANDLING = "threshold_or_negatives (decide at train time)"


def print_guideline() -> None:
    """Pretty-print the annotation guideline (runnable spec check)."""
    print("=" * 72)
    print("ASSISTANT INTENT CLASSIFIER — v1 annotation guideline")
    print("=" * 72)
    print(f"Intents ({len(INTENT_IDS)}): {', '.join(INTENT_IDS)}\n")
    for iid in INTENT_IDS:
        spec = INTENT_SPEC[iid]
        print(f"[{iid}]")
        print(f"  def : {spec['definition']}")
        print(f"  yes : " + " | ".join(spec["examples"]))
        for rule in spec["not"]:
            print(f"  not : {rule}")
        print()
    print("-" * 72)
    print("BOUNDARY RULES (apply when an example could fit two labels):")
    for i, rule in enumerate(BOUNDARY_RULES, 1):
        print(f"  {i}. {rule}")
    print()
    print(f"multi-intent policy : {MULTI_INTENT_POLICY}")
    print(f"none handling       : {NONE_HANDLING}")


if __name__ == "__main__":
    print_guideline()
