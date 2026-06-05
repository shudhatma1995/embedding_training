"""
build.py  -  assemble seeds + templates into honest train/test splits.
================================================================================
The orchestration step, and where the "keep the test set honest" logic lives:
collect every generated string with the intent(s) that produced it, drop
cross-intent label conflicts, and drop any test string that leaked into train.
"""
import random
from collections import defaultdict

from .slots import slots_for, BASE_SLOTS
from .templates import TEMPLATES
from .engine import expand, validate_templates

try:
    from ..intents import INTENT_IDS
    from ..seeds import SEEDS
except ImportError:  # running from inside assistant_intent/ without the package
    from intents import INTENT_IDS
    from seeds import SEEDS


def build_dataset(seed: int = 0, n_train: int = 16, n_test: int = 10):
    validate_templates(TEMPLATES, BASE_SLOTS)
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
