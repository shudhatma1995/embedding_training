"""
eval_wild.py  -  a HAND-WRITTEN, realistic out-of-distribution eval set.
================================================================================
Our data/test.json is template-generated: held-out phrasings + unseen entities,
but still the same clean, synthetic *style* as training. Real users are messier —
casual, indirect, terse, with the intent often implied rather than stated. This
file is hand-written to capture that, so we can measure the gap between
"held-out templates" and "real-ish phrasing" (the honesty gap our template test
can't see).

Design:
  * NOT generated from templates — every line is written to dodge the train
    phrasings on purpose (casual / indirect / fragmentary).
  * Realism is SEMANTIC, not adversarial: no deliberate typos/OOV (those would
    test our word-tokenizer, not the embedding). e.g. "it's too bright in here"
    (-> smart_home), not "tunr off teh lites".
  * `none` is deliberately LARGE and varied (template test had only ~14), both to
    stabilize the τ threshold and because real out-of-scope traffic is huge.
  * Hard/indirect cases are marked  # tricky  with the reasoning, and labeled by
    the DOMINANT reading per intents.py.

Run:  python eval_wild.py        (self-check + write data/test_wild.json)
Use:  it's loaded like test.json — build prototypes from data/train.json, then
      evaluate queries from data/test_wild.json (same evaluate_model harness).
"""

import json
import os

try:
    from .intents import INTENT_IDS
except ImportError:  # running as a plain script from inside assistant_intent/
    from intents import INTENT_IDS

WILD = {
    "answers": [
        "hey how many continents are there",
        "who's the current president of france",
        "is a tomato a fruit or a vegetable",
        "whats the boiling point of water",  # casual, no apostrophe
        "how long do dogs usually live for",
        "what's that movie with the blue aliens called",
        "explain how vaccines work",
        "what does gdp actually stand for",
        "roughly how far is the sun from earth",
        "who wrote romeo and juliet again",
        "remind me how many feet are in a mile",  # tricky: "remind me" but it's a fact lookup -> answers
        "is it true that goldfish have a 3 second memory",
    ],
    "media": [
        "put some music on",
        "i wanna hear something upbeat",
        "can we get some tunes going",
        "throw on a podcast",
        "next one please",  # skip
        "make it a bit louder",
        "ugh skip this song",
        "play that one song from frozen",
        "shuffle everything",
        "pause it for a sec",
        "can you put on some background music",
        "crank it up",
    ],
    "smart_home": [
        "it's too bright in here",  # tricky: implied -> dim the lights
        "i can't see anything in here",  # tricky: implied -> turn lights on
        "kill the lights",
        "can we get some light in here",
        "make it cozy in the living room",  # tricky: implied -> dim / warm light
        "lights down please",
        "it's way too dark",
        "shut the lights off, i'm heading to bed",
        "brighten things up a bit",
        "set the bedroom to something softer",
        "the hallway's pitch black",
        "dim it a little",
    ],
    "productivity": [
        "don't let me forget to call the dentist",  # indirect reminder
        "what've i got going on tomorrow",
        "block out some time friday morning",
        "push my 3 o'clock back an hour",
        "am i busy tonight",
        "jot down a reminder to return the library books",
        "when's my next thing",
        "clear my afternoon",
        "put lunch with sarah on my calendar for thursday",
        "do i have any free time this week",
        "move tomorrow's standup to the afternoon",
        "what's first on my schedule monday",
    ],
    "communication": [
        "ring my mom",
        "shoot dad a quick text",
        "tell sarah i'm running late",  # send a message
        "did anyone email me",
        "get my boss on the phone",
        "drop priya a line",
        "let the team know i'll be there soon",
        "any new messages for me",
        "send grandma a note saying happy birthday",
        "read me my unread emails",
        "facetime my brother",
        "reply to that last email from alex",
    ],
    "timers_alarms": [
        "wake me up in the morning",  # alarm, vague time
        "ping me in 20",  # tricky: 20-min alert, no task -> timer (not a reminder)
        "buzz me when ten minutes is up",
        "i need to be up by 6",
        "give me a countdown from five minutes",
        "kill the timer",
        "how much longer on the timer",
        "set something for half an hour",
        "snooze it",
        "let me know when 15 minutes have passed",
        "wake me before noon",
        "start a two minute timer for the eggs",
    ],
    "weather": [
        "do i need a coat today",  # indirect
        "is it gonna be nice out",
        "should i grab an umbrella",
        "what's it like outside right now",
        "gonna rain later?",
        "how cold is it out there",
        "is it sweater weather",
        "whats the temp gonna be tomorrow",
        "any chance of snow this week",
        "is it nice in miami right now",
        "do i need sunscreen today",
        "how's the weekend looking weather wise",
    ],
    "navigation": [
        "how do i get home from here",
        "is there much traffic on the way",
        "find me a coffee shop nearby",
        "how far is it to the airport",
        "whats the fastest way to the office",
        "where's the closest gas station",
        "give me directions to the new place downtown",
        "how long till i get there",
        "any backups on the highway right now",
        "take me to grandma's house",
        "what's the best route to the stadium",
        "is the bridge backed up",
    ],
    "none": [
        # chit-chat / statements / feelings
        "i'm so bored",
        "ugh i'm exhausted",
        "i think i left the oven on",
        "lol that's hilarious",
        "mmm not sure about that",
        "whatever you say",
        # unsupported actions
        "order takeout for dinner",
        "book a table for two tonight",
        "add eggs to my shopping list",  # shopping list = out of scope
        "open the blinds in the bedroom",  # device, but not lights -> none
        "take a photo",
        "send fifty dollars to john",  # payments -> none
        "set up my new printer",
        "call me an ambulance",  # tricky: "call" but not contacting a person
        "translate good morning into spanish",  # translation -> none
        "play a game of tic tac toe",
        "turn up the thermostat",  # device, not lights -> none
        # subjective / rhetorical near-boundaries
        "what should i wear today",  # tricky: subjective, NOT weather ("do i need a coat" is)
        "remind me why i even bother",  # tricky: rhetorical, NOT a real reminder
        "tell me a joke",  # not a knowledge lookup -> none
        "what's your name",
        "are you a robot",
        # gibberish / noise
        "asdfghjkl",
        "blah blah just testing this",
        "qwerty mcqwerterson",
        "umm hold on",
    ],
}


def _rows():
    return [{"text": t, "intent": iid} for iid in INTENT_IDS for t in WILD[iid]]


def _self_check():
    assert set(WILD.keys()) == set(INTENT_IDS), (
        f"WILD keys {set(WILD)} != INTENT_IDS {set(INTENT_IDS)}"
    )
    seen, clashes, total = {}, [], 0
    print("WILD eval phrasings per intent:")
    for iid in INTENT_IDS:
        lst = WILD[iid]
        total += len(lst)
        dupes = {x for x in lst if lst.count(x) > 1}
        print(f"  {iid:<14} {len(lst):>2}" + (f"  DUPES: {dupes}" if dupes else ""))
        for s in lst:
            if s in seen and seen[s] != iid:
                clashes.append((s, seen[s], iid))
            seen[s] = iid
    print(f"  {'TOTAL':<14} {total:>2}")
    print("  cross-intent clashes:", clashes or "none")
    return total


if __name__ == "__main__":
    n = _self_check()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "data", "test_wild.json")

    # leak check: a wild line that exactly matches a train line would be cheating
    train_path = os.path.join(here, "data", "train.json")
    train_texts = (
        {r["text"] for r in json.load(open(train_path))} if os.path.exists(train_path) else set()
    )
    rows = [r for r in _rows() if r["text"] not in train_texts]  # drop any train leak
    dropped = n - len(rows)
    print(f"\n  exact-overlap with train.json dropped: {dropped}")

    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  wrote {len(rows)} rows -> {out}")
