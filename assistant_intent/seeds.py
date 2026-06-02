"""
seeds.py  -  hand-written SEED phrasings, ~15 per intent.
========================================================

These are starting points, not the final training set. In the next step
(make_data.py) we turn patterns here into templates + slot lists to generate
hundreds of varied examples. Every line is labelled per the rules in intents.py.

Deliberate design choices baked in:
  * diversity of register: formal, casual, terse ("lights off"), slang ("ugh").
  * NEAR-COLLISIONS on purpose (marked #collision) -- pairs that share words but
    mean different things, e.g. "turn on the lights" (smart_home) vs
    "turn on some music" (media), "play some jazz" (media) vs "play with my dog"
    (none). These are what force the model to learn meaning, not keywords.
  * productivity covers BOTH calendar and email, roughly balanced, so the bundle
    can later split cleanly.
  * none includes gibberish, statements, and out-of-scope requests -- including
    things that are *almost* a supported intent but outside v1 scope
    (timers, garage door, shopping list).
All lowercase (the tokenizer lowercases anyway; keeps seeds consistent).
"""

try:
    from .intents import INTENT_IDS
except ImportError:  # running as a plain script from inside assistant_intent/
    from intents import INTENT_IDS

SEEDS = {
    "answers": [
        "how tall is mount everest",
        "what's 15 percent of 80",
        "who won the 2022 world cup",
        "what time is it in tokyo",                 # collision: time q but NO app -> answers
        "how do you say thank you in french",
        "what's the capital of australia",
        "how many ounces are in a pound",
        "why is the sky blue",
        "who painted the mona lisa",
        "how long does it take to boil an egg",
        "what year did the berlin wall fall",
        "convert 10 miles to kilometers",
        "what's the square root of 144",
        "how far is the moon from earth",
        "what's the difference between weather and climate",
    ],
    "media": [
        "play some jazz",
        "pause the music",
        "skip to the next track",
        "turn it up",
        "play the latest taylor swift album",
        "put on my workout playlist",
        "resume my podcast",
        "play the office on netflix",
        "shuffle my liked songs",
        "turn the volume down a bit",
        "go back to the previous song",
        "play something relaxing",
        "stop the music",
        "play lofi beats on spotify",
        "turn on some music",                       # collision: "turn on" + music -> media
    ],
    "smart_home": [
        "turn on the living room lights",
        "lights off",
        "dim the bedroom lights to 50 percent",
        "switch off the kitchen light",
        "turn on the lights",                       # collision: "turn on" + lights -> smart_home
        "make it brighter in here",
        "turn off all the lights",
        "set the lights to warm white",
        "can you turn on the porch light",
        "dim the lights a little",
        "switch on the lamp in the office",
        "lights on please",
        "turn the hallway light off",
        "brighten the lights",
        "turn on the bedroom lamp",
    ],
    "productivity": [
        # --- calendar ---
        "schedule a meeting with john tomorrow at 3",
        "what's on my calendar today",              # collision: question, but needs app -> productivity
        "add a dentist appointment on friday",
        "remind me to call the bank",
        "move my 2pm meeting to 4",
        "am i free thursday afternoon",
        "set up a calendar event for the standup",
        # --- email ---
        "email mom that i'll be late",
        "send an email to my boss about the report",
        "do i have any new emails",
        "reply to sarah's email",
        "read me my latest email",
        "compose an email to the team",
        "check my inbox",
        "forward that email to alex",
    ],
    "none": [
        "asdfgh",
        "i'm so tired today",
        "book me a flight to paris",
        "order me a pizza",
        "ugh what a day",
        "call me an uber",
        "set a 10 minute timer",                    # almost an assistant feature, but out of v1 scope
        "translate this sentence into spanish",
        "play with my dog",                         # collision: "play" but NOT media -> none
        "i think i left the stove on",
        "take a selfie",
        "open the garage door",                     # collision: device, but not lights -> none (v1 scope)
        "add milk to my shopping list",             # collision: "add ..." but not calendar/email -> none
        "mmm okay sure",
        "what should i cook for dinner",            # subjective/open, not a knowledge lookup -> none
    ],
}


def _self_check() -> None:
    # every key must be a known intent, and vice versa
    assert set(SEEDS.keys()) == set(INTENT_IDS), (
        f"SEEDS keys {set(SEEDS)} != INTENT_IDS {set(INTENT_IDS)}"
    )
    print("SEED phrasings per intent:")
    total = 0
    for iid in INTENT_IDS:
        n = len(SEEDS[iid])
        total += n
        # flag duplicates within an intent
        dupes = [x for x in SEEDS[iid] if SEEDS[iid].count(x) > 1]
        flag = f"  DUPLICATES: {set(dupes)}" if dupes else ""
        print(f"  {iid:<13} {n:>2}{flag}")
    print(f"  {'TOTAL':<13} {total:>2}")
    # cross-intent duplicate check (same string in two intents = contradiction)
    seen = {}
    clash = []
    for iid, lst in SEEDS.items():
        for s in lst:
            if s in seen and seen[s] != iid:
                clash.append((s, seen[s], iid))
            seen[s] = iid
    print("  cross-intent clashes:", clash if clash else "none")


if __name__ == "__main__":
    _self_check()
