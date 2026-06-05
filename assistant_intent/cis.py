"""
cis.py  -  borrow the embedder + tokenizer from customer_intent_search, once.
================================================================================
`customer_intent_search/` is reused UNTOUCHED so the upstream tutorial stays cleanly
mergeable. Importing this module puts that directory on sys.path and re-exports the
two pieces we need. Centralized here so train.py / evaluate.py don't each copy the
sys.path shim — the shim (and its ordering hazard) now lives in exactly ONE place.

Ordering hazard: this inserts customer_intent_search/ at the FRONT of sys.path, and
that directory ALSO has an evaluate.py importing sentence_transformers. Our sibling
modules are named so nothing there shadows them, and anything that must load our own
evaluate.py by name (experiments.py) does so by explicit path.
"""
import os
import sys

_CIS = os.path.join(os.path.dirname(__file__), "..", "customer_intent_search")
if _CIS not in sys.path:
    sys.path.insert(0, _CIS)

from tokenizer import SimpleTokenizer   # noqa: E402  (resolved from customer_intent_search)
from model import build_model           # noqa: E402

__all__ = ["SimpleTokenizer", "build_model"]
