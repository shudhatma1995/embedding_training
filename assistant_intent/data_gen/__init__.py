"""
data_gen  -  Stage 2: turn seeds into a real dataset via template expansion.
================================================================================
(Split out of the old single-file `make_data.py`. Same behaviour, four concerns
now live in four files so the data you edit most -- slots and templates -- no
longer buries the ~85 lines of logic that actually does the work.)

Why templates + slots (not just hand-writing 600 sentences):
  A template like "turn {on|off} the {room} {fixture}" x slot lists generates
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

Layout:
    slots.py     - BASE_SLOTS, ENTITY_SLOTS, slots_for()   (slot value pools)
    templates.py - TEMPLATES                               (per-intent phrasings)
    engine.py    - fill(), expand(), validate_templates()  (expansion machinery)
    build.py     - build_dataset()                         (assembly + leak/conflict filtering)
    __main__.py  - CLI + reporting

Run:   python -m assistant_intent.data_gen   (writes data/train.json, data/test.json)
Use:   from assistant_intent.data_gen import build_dataset
"""
from .build import build_dataset
from .slots import BASE_SLOTS, ENTITY_SLOTS, slots_for
from .templates import TEMPLATES
from .engine import fill, expand, validate_templates

__all__ = [
    "build_dataset",
    "BASE_SLOTS",
    "ENTITY_SLOTS",
    "slots_for",
    "TEMPLATES",
    "fill",
    "expand",
    "validate_templates",
]
