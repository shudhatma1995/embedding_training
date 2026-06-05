"""
__main__.py  -  CLI: build the dataset, write json, print a readable report.
================================================================================
Run:  python -m assistant_intent.data_gen            (writes ../data/train.json, ../data/test.json)
"""
import os
import json
import random
import argparse
from collections import Counter

from .build import build_dataset

try:
    from ..intents import INTENT_IDS
except ImportError:  # running from inside assistant_intent/ without the package
    from intents import INTENT_IDS

# default output dir is assistant_intent/data (one level up from this package)
_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _stats(rows, title):
    c = Counter(r["intent"] for r in rows)
    print(f"  {title} ({len(rows)} total)")
    for iid in INTENT_IDS:
        print(f"    {iid:<13} {c.get(iid, 0):>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=_DEFAULT_OUT)
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
