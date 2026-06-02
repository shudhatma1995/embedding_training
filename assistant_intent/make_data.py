"""
make_data.py  -  Stage 2: turn seeds into a real dataset via template expansion.
================================================================================

Why templates + slots (not just hand-writing 600 sentences):
  A template like "turn {on|off} the {room} {fixture}" × slot lists generates
  hundreds of varied sentences cheaply. Crucially, the ENTITY values (rooms,
  artists, contacts...) vary a lot, which stops the model from treating a
  specific word ("bedroom") as the intent signal -- it must learn the PATTERN.

How we keep the test set honest (no leakage):
  1. TEMPLATE split    - test uses DIFFERENT phrasings than train.
  2. ENTITY split      - some slot values (e.g. certain rooms/artists) appear
                         ONLY in test, so test sentences contain unseen words.
  3. Overlap filter    - any test string that still collides with a train
                         string is dropped.
  4. Conflict check    - if the same string is generated for two intents, that
                         is a labelling contradiction -> dropped + reported.

Run:   python make_data.py            (writes data/train.json, data/test.json)
"""
import os
import re
import json
import random
import argparse
from collections import Counter, defaultdict

try:
    from .intents import INTENT_IDS
    from .seeds import SEEDS
except ImportError:  # running as a plain script from inside assistant_intent/
    from intents import INTENT_IDS
    from seeds import SEEDS

# ── slot value pools ────────────────────────────────────────────────────────
BASE_SLOTS = {
    "room":     ["living room", "bedroom", "kitchen", "hallway", "office",
                 "bathroom", "dining room", "garage", "basement", "porch",
                 "nursery", "study"],
    "fixture":  ["light", "lights", "lamp", "ceiling light"],
    "pct":      ["10", "20", "30", "40", "50", "60", "70", "80"],
    "genre":    ["jazz", "classical music", "lofi beats", "hip hop", "rock",
                 "country", "pop", "edm", "blues", "reggae", "ambient music",
                 "k-pop"],
    "artist":   ["taylor swift", "drake", "the beatles", "beyonce", "adele",
                 "coldplay", "kendrick lamar", "billie eilish", "ed sheeran",
                 "bad bunny", "sza", "the weeknd"],
    "playlist": ["my workout playlist", "my chill playlist", "my focus playlist",
                 "my road trip playlist", "my liked songs",
                 "my dinner party playlist"],
    "service_phrase": ["on spotify", "on apple music", "on youtube music",
                       "on soundcloud", ""],
    "contact":  ["john", "sarah", "mom", "my boss", "alex", "dad", "the team",
                 "my landlord", "grandma", "dr patel", "mike", "priya"],
    "when_phrase": ["tomorrow", "on monday", "on friday", "at 3", "at 4pm",
                    "next week", "this afternoon", "on saturday", ""],
    "time":     ["3pm", "4pm", "9am", "noon", "2:30", "5 o'clock", "10am", "1pm"],
    "reminders": ["call the bank", "pick up groceries", "water the plants",
                  "send the invoice", "feed the cat", "book the venue",
                  "take out the trash", "renew my passport"],
    "country":  ["france", "japan", "australia", "brazil", "canada", "egypt",
                 "kenya", "peru", "italy", "india", "norway", "chile"],
    "language": ["french", "spanish", "german", "japanese", "hindi", "italian",
                 "mandarin"],
    "word":     ["thank you", "hello", "goodbye", "please", "water", "friend"],
    "conversion_phrase": ["10 miles to kilometers", "5 kg to pounds",
                          "100 dollars to euros", "3 cups to milliliters",
                          "20 celsius to fahrenheit"],
    "landmark": ["mount everest", "the eiffel tower", "the great wall",
                 "the nile", "the pacific ocean", "mount fuji",
                 "the grand canyon", "the sahara"],
    "achievement": ["won the 2022 world cup", "painted the mona lisa",
                    "wrote hamlet", "invented the telephone",
                    "discovered penicillin", "composed the ninth symphony"],
    "event":    ["the berlin wall fall", "world war two end",
                 "the first moon landing happen", "the titanic sink"],
    "phenomenon": ["the sky blue", "the ocean salty", "grass green",
                   "the moon visible during the day"],
    "food":     ["pizza", "sushi", "a burger", "tacos", "coffee", "thai food",
                 "a sandwich", "ramen"],
    "transport": ["flight", "taxi", "uber", "train ticket", "cab", "rideshare"],
    "n":        ["5", "10", "15", "20", "30"],
    "n2":       ["40", "60", "80", "100", "200"],
    "gibberish": ["asdfgh", "blah blah blah", "qwerty asdf", "lorem ipsum",
                  "mmm hmm", "uhh what"],
}

# These high-variety slots get split: ~70% of values for train, ~30% for test
# (test sentences therefore contain ENTITIES the model never saw in training).
ENTITY_SLOTS = {"room", "genre", "artist", "playlist", "contact", "country",
                "landmark", "food", "transport"}

# ── templates per intent: separate phrasings for train vs test ───────────────
TEMPLATES = {
    "smart_home": {
        "train": [
            "turn {on|off} the {room} {fixture}",
            "{switch on|switch off|turn on|turn off} the {fixture} in the {room}",
            "{fixture} {on|off}",
            "dim the {room} {fixture} to {pct} percent",
            "make the {room} {brighter|darker}",
            "set the {room} {fixture} to {warm white|cool white|daylight}",
        ],
        "test": [
            "can you {turn on|turn off} the {room} {fixture}",
            "{brighten|dim} the {fixture} in the {room}",
            "i want the {room} {fixture} {on|off}",
        ],
    },
    "media": {
        "train": [
            "play {genre}",
            "play {artist}",
            "put on {playlist}",
            "play some {genre} {service_phrase}",
            "{pause|stop|resume} the {music|podcast|song}",
            "skip {to the next track|this song|ahead}",
            "turn the volume {up|down}",
            "play {playlist} {service_phrase}",
        ],
        "test": [
            "i want to listen to {genre}",
            "can you play {artist} {service_phrase}",
            "go to the {next|previous} {song|track}",
            "turn it {up|down}",
        ],
    },
    "productivity": {
        "train": [
            "schedule a {meeting|call} with {contact} {when_phrase}",
            "add {a dentist appointment|a meeting|a reminder} {when_phrase}",
            "what's on my calendar {today|tomorrow|this week}",
            "am i free {when_phrase}",
            "remind me to {reminders}",
            "move my {time} {meeting|call} to {time}",
            "email {contact} {about the report|that i'll be late|the meeting notes}",
            "send {contact} an email",
            "{reply to|forward} {contact}'s email",
            "read me my {latest|last} email",
            "check my inbox",
        ],
        "test": [
            "set up a {meeting|call} with {contact} {when_phrase}",
            "do i have anything {today|tomorrow}",
            "write an email to {contact}",
            "any new emails from {contact}",
            "set a reminder to {reminders}",
        ],
    },
    "answers": {
        "train": [
            "what's the capital of {country}",
            "how do you say {word} in {language}",
            "convert {conversion_phrase}",
            "what's {n} percent of {n2}",
            "who {achievement}",
            "how {tall|deep|far|big} is {landmark}",
            "why is {phenomenon}",
        ],
        "test": [
            "tell me the capital of {country}",
            "what year did {event}",
            "what's the {height|size} of {landmark}",
            "how do i say {word} in {language}",
        ],
    },
    "none": {
        "train": [
            "order me {food}",
            "book me a {transport}",
            "set a {n} minute timer",
            "call me {an uber|a taxi|a cab}",
            "add {food} to my shopping list",
            "{ugh|meh|hmm} {what a day|whatever|never mind}",
            "{gibberish}",
        ],
        "test": [
            "i need a {transport} to the airport",
            "order {food} for delivery",
            "start a {timer|stopwatch} for {n} minutes",
            "translate this {document|page} into {language}",
        ],
    },
}

_TOKEN = re.compile(r"\{([^{}]+)\}")


def slots_for(split: str) -> dict:
    """Return slot pools for a split; entity slots are partitioned 70/30."""
    out = {}
    for name, vals in BASE_SLOTS.items():
        if name in ENTITY_SLOTS:
            k = max(1, round(len(vals) * 0.7))
            out[name] = vals[:k] if split == "train" else vals[k:]
        else:
            out[name] = vals
    return out


def fill(template: str, slots: dict, rng: random.Random) -> str:
    """Replace {slot} (lookup) or {a|b|c} (inline choice) tokens; tidy spaces."""
    def repl(m):
        body = m.group(1)
        if "|" in body:
            return rng.choice(body.split("|"))
        return rng.choice(slots[body])
    return re.sub(r"\s+", " ", _TOKEN.sub(repl, template)).strip()


def expand(template: str, slots: dict, rng: random.Random, k: int) -> list:
    """Up to k UNIQUE fillings of a template (fewer if combos run out)."""
    seen, out, attempts = set(), [], 0
    while len(out) < k and attempts < k * 12:
        attempts += 1
        s = fill(template, slots, rng)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def validate_templates() -> None:
    """Every {slot} (non-inline) token must exist in BASE_SLOTS."""
    for intent, splits in TEMPLATES.items():
        for split, tmpls in splits.items():
            for t in tmpls:
                for body in _TOKEN.findall(t):
                    if "|" in body:
                        continue
                    assert body in BASE_SLOTS, \
                        f"unknown slot '{{{body}}}' in {intent}/{split}: {t!r}"


def build_dataset(seed: int = 0, n_train: int = 16, n_test: int = 10):
    validate_templates()
    rng = random.Random(seed)
    st, sv = slots_for("train"), slots_for("test")

    # collect (text -> set of intents) so we can detect cross-intent conflicts
    train_map, test_map = defaultdict(set), defaultdict(set)

    for intent in INTENT_IDS:
        for s in SEEDS[intent]:                      # seeds are natural train data
            train_map[s].add(intent)
        for t in TEMPLATES[intent]["train"]:
            for s in expand(t, st, rng, n_train):
                train_map[s].add(intent)
        for t in TEMPLATES[intent]["test"]:
            for s in expand(t, sv, rng, n_test):
                test_map[s].add(intent)

    # drop label conflicts (same text, two intents) and report them
    conflicts = {s: ints for d in (train_map, test_map)
                 for s, ints in d.items() if len(ints) > 1}
    train = {s: next(iter(i)) for s, i in train_map.items() if len(i) == 1}
    test  = {s: next(iter(i)) for s, i in test_map.items()  if len(i) == 1}

    # leakage: any test string also in train -> remove from test
    leaked = [s for s in test if s in train]
    for s in leaked:
        del test[s]

    train_rows = [{"text": s, "intent": i} for s, i in train.items()]
    test_rows  = [{"text": s, "intent": i} for s, i in test.items()]
    return train_rows, test_rows, conflicts, leaked


def _stats(rows, title):
    c = Counter(r["intent"] for r in rows)
    print(f"  {title} ({len(rows)} total)")
    for iid in INTENT_IDS:
        print(f"    {iid:<13} {c.get(iid, 0):>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    args = ap.parse_args()

    train, test, conflicts, leaked = build_dataset(seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(args.out, "test.json"), "w") as f:
        json.dump(test, f, indent=2)

    print("=" * 60)
    print("DATASET BUILT")
    print("=" * 60)
    _stats(train, "TRAIN")
    print()
    _stats(test, "TEST  (unseen phrasings + unseen entities)")
    print()
    print(f"  label conflicts dropped : {len(conflicts)}", conflicts or "")
    print(f"  test->train leaks dropped: {len(leaked)}")
    print(f"\n  wrote {args.out}/train.json  and  /test.json")
    # a few samples so you can eyeball quality
    rng = random.Random(123)
    print("\n  sample TEST rows:")
    for r in rng.sample(test, min(8, len(test))):
        print(f"    [{r['intent']:<12}] {r['text']}")


if __name__ == "__main__":
    main()
