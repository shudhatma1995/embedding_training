"""Shared pytest setup + fixtures for the assistant_intent suite.

Import plumbing
---------------
The package uses bare imports (`from data import ...`, `from cis import ...`) and
`cis.py` inserts customer_intent_search/ at the FRONT of sys.path. Both packages
also contain an `evaluate.py` and a `train.py`, so a naive `import evaluate` after
cis has run would resolve to the WRONG (customer_intent_search) module.

We sidestep that by:
  1. putting both package dirs on sys.path (unique module names resolve fine), and
  2. eagerly loading the two *colliding* modules from their assistant_intent paths
     and registering them in sys.modules under their bare names, so any later
     `import evaluate` / `import train` returns the assistant_intent version.
"""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSISTANT = os.path.join(ROOT, "assistant_intent")
CIS = os.path.join(ROOT, "customer_intent_search")

for _p in (ASSISTANT, CIS):
    if _p not in sys.path:
        sys.path.append(_p)


def _load_assistant_module(name, filename):
    """Load assistant_intent/<filename> as the bare module `name` and cache it,
    so the customer_intent_search namesake can never shadow it."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ASSISTANT, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec so internal imports see it
    spec.loader.exec_module(mod)
    return mod


# Pin the two collision-prone modules to their assistant_intent versions.
_load_assistant_module("evaluate", "evaluate.py")
_load_assistant_module("train", "train.py")
