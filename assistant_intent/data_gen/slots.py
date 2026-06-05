"""
slots.py  -  slot value pools + the train/test entity split.
================================================================================
Pure data, plus one helper. A {slot} token in a template is replaced by a random
value from the matching list here. ENTITY_SLOTS are high-variety pools (rooms,
artists, contacts...) that we partition 70/30 so test sentences contain entities
the model never saw in training.
"""

# slot value pools
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
    # timers_alarms
    "duration": ["10 minutes", "5 minutes", "20 minutes", "15 minutes",
                 "30 minutes", "2 minutes", "45 minutes", "1 hour",
                 "90 seconds", "3 minutes"],
    "timer_label": ["the pasta", "the laundry", "the oven", "my workout",
                    "the tea", "the eggs"],
    # navigation (place is an entity slot → 70/30 so test has unseen destinations)
    "place": ["the airport", "downtown", "home", "work", "the grocery store",
              "the mall", "the train station", "the nearest pharmacy",
              "the beach", "the stadium", "the hospital", "the nearest gas station"],
    # weather (city is an entity slot)
    "city": ["new york", "london", "paris", "tokyo", "chicago", "seattle",
             "miami", "denver"],
    # communication
    "message_topic": ["about the report", "that i'll be late", "the meeting notes",
                      "to say happy birthday", "about dinner", "that i'm on my way"],
}

# These high-variety slots get split: ~70% of values for train, ~30% for test
# (test sentences therefore contain ENTITIES the model never saw in training).
ENTITY_SLOTS = {"room", "genre", "artist", "playlist", "contact", "country",
                "landmark", "food", "transport", "place", "city"}


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
