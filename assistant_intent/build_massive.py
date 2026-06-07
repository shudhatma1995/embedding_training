"""
build_massive.py  -  fetch Amazon MASSIVE and project it onto OUR taxonomy.
================================================================================

Produces data/test_massive.json — a TRUE external test set (real, third-party,
human-collected utterances) in our {text, intent} format, so it drops straight
into the existing evaluate_model harness exactly like test.json / test_wild.json.

This is the one experiment every result so far has dodged: our synthetic test and
our hand-written wild set were both authored by us, so they only prove "generalize
to OUR style". MASSIVE was not — so it finally tests generalization to real users.

Design choices (infra notes):
  - No new dependency. We pull rows from the Hugging Face datasets-server REST API
    over stdlib urllib (the heavyweight `datasets` lib is unnecessary for a
    run-once fetch). Intent ints are decoded via the dataset's ClassLabel names.
  - The label decision lives entirely in massive_mapping.py (pure + unit-tested);
    this script only fetches, filters, balances, and writes.
  - The output is NOT committed (it is third-party CC-BY-4.0 data); it is gitignored
    and regenerated on demand. The committed, reviewed artifacts are the mapping,
    this builder, and the eval — not the corpus.
  - Leakage + balance: we drop any utterance that exactly matches a train.json text,
    dedupe, and cap per OUR-intent (seeded) so no intent dominates the metrics.

Source: AmazonScience/massive (CC-BY-4.0), https://huggingface.co/datasets/AmazonScience/massive
        FitzGerald et al., "MASSIVE: A 1M-Example Multilingual NLU Dataset" (2022).

Run:  python build_massive.py            (writes data/test_massive.json + .meta.json)
      python build_massive.py --cap-per-intent 80 --seed 1
"""

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request

from data import DATA_DIR, load_by_intent
from intents import NONE_ID, REAL_INTENTS
from massive_mapping import map_intent

DATASET = "AmazonScience/massive"
CONFIG = "en-US"
API = "https://datasets-server.huggingface.co"
PAGE = 100  # datasets-server max rows per request


def _get_json(url, retries=4, backoff=1.5):
    """GET a JSON document with a few retries on transient network/5xx errors."""
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:  # pragma: no cover - network
            last = e
            time.sleep(backoff**attempt)
    raise RuntimeError(f"failed to GET {url}: {last}")


def _intent_names():
    """The ordered ClassLabel names for `intent`, used to decode the integer codes."""
    info = _get_json(f"{API}/info?dataset={urllib.parse.quote(DATASET)}&config={CONFIG}")
    return info["dataset_info"]["features"]["intent"]["names"]


def fetch_rows(split):
    """Yield (massive_intent, utterance) for every row in the split, depaginated."""
    names = _intent_names()
    base = f"{API}/rows?dataset={urllib.parse.quote(DATASET)}&config={CONFIG}&split={split}"
    offset = 0
    while True:
        page = _get_json(f"{base}&offset={offset}&length={PAGE}")
        rows = page.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            yield names[row["intent"]], row["utt"]
        offset += len(rows)
        if offset >= page.get("num_rows_total", offset):
            break


def build(split, cap_per_intent, none_cap, seed):
    """Fetch → map → drop-leaks → dedupe → balance. Returns (rows, meta)."""
    train_texts = set()
    for texts in load_by_intent(
        os.path.join(DATA_DIR, "train.json"), [*REAL_INTENTS, NONE_ID]
    ).values():
        train_texts.update(t.strip().lower() for t in texts)

    # collect mapped, de-leaked, de-duped utterances bucketed by OUR intent
    by_intent = {it: [] for it in [*REAL_INTENTS, NONE_ID]}
    seen = set()
    n_fetched = n_dropped_unmapped = n_dropped_leak = n_dropped_dup = 0
    for massive_intent, utt in fetch_rows(split):
        n_fetched += 1
        label = map_intent(massive_intent)
        if label is None:
            n_dropped_unmapped += 1
            continue
        norm = utt.strip().lower()
        if norm in train_texts:
            n_dropped_leak += 1
            continue
        if norm in seen:
            n_dropped_dup += 1
            continue
        seen.add(norm)
        by_intent[label].append(utt)

    # balance: seeded sample per intent (real intents capped; none gets its own cap)
    rng = random.Random(seed)
    rows, per_intent = [], {}
    for it in [*REAL_INTENTS, NONE_ID]:
        pool = by_intent[it]
        rng.shuffle(pool)
        cap = none_cap if it == NONE_ID else cap_per_intent
        kept = pool[:cap]
        per_intent[it] = len(kept)
        rows.extend({"text": t, "intent": it} for t in kept)
    rng.shuffle(rows)

    meta = {
        "source": DATASET,
        "source_url": "https://huggingface.co/datasets/AmazonScience/massive",
        "license": "CC-BY-4.0",
        "citation": "FitzGerald et al., MASSIVE (2022), arXiv:2204.08582",
        "config": CONFIG,
        "split": split,
        "params": {"cap_per_intent": cap_per_intent, "none_cap": none_cap, "seed": seed},
        "fetched": n_fetched,
        "dropped": {
            "unmapped_or_ambiguous": n_dropped_unmapped,
            "train_leak": n_dropped_leak,
            "duplicate": n_dropped_dup,
        },
        "kept_total": len(rows),
        "kept_per_intent": per_intent,
    }
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description="Build the MASSIVE external test set")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--cap-per-intent", type=int, default=60, help="max queries per REAL intent")
    ap.add_argument("--none-cap", type=int, default=120, help="max none queries")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "test_massive.json"))
    args = ap.parse_args()

    print(f"Fetching {DATASET} [{CONFIG}] split={args.split} via datasets-server ...")
    rows, meta = build(args.split, args.cap_per_intent, args.none_cap, args.seed)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    with open(args.out.replace(".json", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  fetched {meta['fetched']} rows; dropped {meta['dropped']}")
    print(f"  wrote {meta['kept_total']} rows -> {args.out}")
    print("  per intent: " + ", ".join(f"{k}={v}" for k, v in meta["kept_per_intent"].items()))


if __name__ == "__main__":
    main()
