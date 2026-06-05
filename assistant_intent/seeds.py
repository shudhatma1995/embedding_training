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
  * communication groups call/text/email (reach a person); productivity is
    calendar + reminders. (Email was split out of productivity once communication
    was added -- a supported capability graduating into its own intent.)
  * none includes gibberish, statements, and out-of-scope requests -- including
    things that are *almost* a supported intent but outside current scope
    (garage door, shopping list, ride-booking). Note: timers used to live here
    and graduated into `timers_alarms`.
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
        "i feel like listening to some jazz",       # desire form (helps "i want to listen to ...")
        "i'd like to hear the beatles",
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
        # calendar + reminders/tasks (email moved to `communication`)
        "schedule a meeting with john tomorrow at 3",
        "what's on my calendar today",              # collision: question, but needs app -> productivity
        "add a dentist appointment on friday",
        "remind me to call the bank",
        "move my 2pm meeting to 4",
        "am i free thursday afternoon",
        "set up a calendar event for the standup",
        "what's my next meeting",
        "cancel my 3pm meeting",
        "remind me to pick up groceries",
        "do i have anything scheduled for monday",
        "block off an hour tomorrow afternoon",
        "when is my next appointment",
        "add lunch with sarah on friday",
        "set a reminder to water the plants",
    ],
    "communication": [
        # --- email (moved out of productivity) ---
        "email mom that i'll be late",
        "send an email to my boss about the report",
        "do i have any new emails",
        "reply to sarah's email",
        "read me my latest email",
        "compose an email to the team",
        "check my inbox",
        "forward that email to alex",
        # --- calls / texts ---
        "call mom",
        "give dad a call",
        "text sarah that i'm on my way",
        "send a text to john",
        "call my boss",
        "facetime grandma",
        "send a message to the team",
    ],
    "timers_alarms": [
        "set a timer for 10 minutes",
        "set a 5 minute timer",
        "set an alarm for 7am",
        "wake me up at 6:30",
        "set an alarm for 6 am tomorrow",
        "cancel my timer",
        "how much time is left on my timer",
        "set a timer for 20 minutes",
        "pause the timer",
        "set an alarm for noon",
        "snooze the alarm",
        "turn off the alarm",                       # collision: "turn off" but alarm, not lights -> timers_alarms
        "start a 30 second timer",
        "set a timer for the pasta",
        "wake me at 8 in the morning",
    ],
    "weather": [
        "what's the weather today",
        "is it going to rain tomorrow",
        "what's the temperature outside",
        "do i need an umbrella today",
        "what's the forecast for this weekend",
        "how hot is it going to be tomorrow",
        "is it cold outside",
        "will it snow tonight",
        "what's the weather like in london",
        "is it sunny right now",
        "how windy is it today",
        "what's the humidity like",
        "should i wear a jacket today",
        "is it going to be nice this weekend",
        "what's the high for today",
    ],
    "navigation": [
        "navigate home",
        "give me directions to the airport",
        "how long will it take to get downtown",
        "what's the fastest route to work",
        "is there traffic on the way to the office",
        "find a gas station near me",
        "take me to the nearest pharmacy",
        "how far is the grocery store",             # collision: "how far" but a reachable place -> navigation
        "directions to the train station",
        "what's the best route home",
        "how's traffic right now",
        "find parking near the stadium",
        "navigate to the mall",
        "how long to the beach",
        "show me the way to the hospital",
    ],
    "none": [
        "asdfgh",
        "i'm so tired today",
        "book me a flight to paris",
        "order me a pizza",
        "ugh what a day",
        "call me an uber",                          # collision: "call" but not a person -> none (ride booking)
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
