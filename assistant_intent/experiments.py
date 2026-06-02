"""
experiments.py  -  Stage 5b: kill the init confound with a multi-seed ablation.
================================================================================

Why this exists
  In Stage 5 we compared single models and noticed that changing the vocabulary
  also changed `vocab_size`, which (with a fixed seed) reshuffles the ENTIRE
  network's random init — a hidden confound. The fix: run each config across
  several seeds and AVERAGE. Init luck and pairing randomness then wash out, and
  the remaining difference is attributable to the config itself.

Configs (everything else identical)
  A  real-only vocab,  no negatives   — junk words become [UNK]
  C  none-in-vocab,    no negatives   — isolates putting junk words in-vocab
  B  none-in-vocab,    k=8 negatives  — adds the none-as-negatives objective

Clean comparisons
  A vs C : does putting junk in-vocab actually hurt? (now seed-averaged, so the
           vocab_size→init reshuffle no longer fakes a difference)
  C vs B : does the negative objective help? (same vocab_size → already clean;
           seeds just confirm it's robust)

Run:  python experiments.py                 (5 seeds, ~2 min on CPU)
      python experiments.py --seeds 0 1 2    (fewer/more seeds)
"""
import os
import io
import shutil
import argparse
import contextlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_local(modname, filename):
    """Load a sibling module by explicit path. Needed because train.py puts
    customer_intent_search/ on sys.path, and its evaluate.py would otherwise
    shadow our local evaluate.py (and drag in sentence_transformers)."""
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load_local("ai_train", "train.py")
ev = _load_local("ai_evaluate", "evaluate.py")

CONFIGS = [
    {"name": "A real-only-vocab", "none_in_vocab": False, "none_neg_k": 0},
    {"name": "C none-in-vocab",   "none_in_vocab": True,  "none_neg_k": 0},
    {"name": "B +negatives k=8",  "none_in_vocab": True,  "none_neg_k": 8},
]


def run_one(cfg, seed, device, train_groups, test_groups, none_texts, scratch):
    """Train one config at one seed (output suppressed) and return eval metrics."""
    args = SimpleNamespace(
        epochs=25, pairs_per_intent=30, lr=3e-4, temperature=0.05,
        warmup_steps=30, seed=seed,
        none_neg_k=cfg["none_neg_k"], none_in_vocab=cfg["none_in_vocab"],
        output_dir=scratch,
    )
    with contextlib.redirect_stdout(io.StringIO()):     # silence the epoch table
        model, tok = tr.train(args)

    micro, _, _, (protos, intents) = ev.evaluate_real(
        model, tok, train_groups, test_groups, device)
    real_max, none_max, _, best, _ = ev.evaluate_none(
        model, tok, test_groups, none_texts, protos, intents, device)
    return {
        "r1": micro["recall@1"],
        "real_sim": float(real_max.mean()),
        "none_sim": float(none_max.mean()),
        "overall": best["overall"],
        "none_recall": best["none_recall"],
    }


def ms(vals, pct=False):
    """Format a list as 'mean±std' (as percentages if pct)."""
    a = np.array(vals)
    if pct:
        return f"{a.mean()*100:4.1f}±{a.std()*100:3.1f}"
    return f"{a.mean():.2f}±{a.std():.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scratch = os.path.join(HERE, "models", "_ablation_tmp")

    train_groups = ev.load_grouped(os.path.join(ev.DATA_DIR, "train.json"), ev.REAL_INTENTS)
    test_groups = ev.load_grouped(os.path.join(ev.DATA_DIR, "test.json"), ev.REAL_INTENTS)
    none_texts = ev.load_grouped(os.path.join(ev.DATA_DIR, "test.json"), ["none"])["none"]

    print("=" * 78)
    print(f"MULTI-SEED ABLATION  (seeds={args.seeds}, {len(args.seeds)} runs per config)")
    print("=" * 78)

    results = {c["name"]: [] for c in CONFIGS}
    for seed in args.seeds:
        for cfg in CONFIGS:
            m = run_one(cfg, seed, device, train_groups, test_groups, none_texts, scratch)
            results[cfg["name"]].append(m)
            print(f"  seed {seed} | {cfg['name']:<20} "
                  f"R@1 {m['r1']*100:5.1f}%  none_sim {m['none_sim']:.2f}  "
                  f"overall {m['overall']*100:5.1f}%")

    shutil.rmtree(scratch, ignore_errors=True)

    cols = [("r1", "4-int R@1", True), ("real_sim", "real sim", False),
            ("none_sim", "none sim", False), ("overall", "best 5-way", True),
            ("none_recall", "none recall", True)]
    print("\n" + "=" * 78)
    print(f"  MEAN ± STD over {len(args.seeds)} seeds")
    print("=" * 78)
    print(f"  {'config':<20}" + "".join(f"{lbl:>14}" for _, lbl, _ in cols))
    for name, runs in results.items():
        row = f"  {name:<20}"
        for key, _, pct in cols:
            row += f"{ms([r[key] for r in runs], pct):>14}"
        print(row)

    # the two clean verdicts
    def mean(name, key):
        return float(np.mean([r[key] for r in results[name]]))
    A, C, B = "A real-only-vocab", "C none-in-vocab", "B +negatives k=8"
    print("\n  verdicts:")
    print(f"    A vs C (vocab effect)   : 4-int R@1 {mean(A,'r1')*100:.1f}% → {mean(C,'r1')*100:.1f}%, "
          f"none_sim {mean(A,'none_sim'):.2f} → {mean(C,'none_sim'):.2f}, "
          f"5-way {mean(A,'overall')*100:.1f}% → {mean(C,'overall')*100:.1f}%")
    print(f"    C vs B (negatives effect): 4-int R@1 {mean(C,'r1')*100:.1f}% → {mean(B,'r1')*100:.1f}%, "
          f"none_sim {mean(C,'none_sim'):.2f} → {mean(B,'none_sim'):.2f}, "
          f"5-way {mean(C,'overall')*100:.1f}% → {mean(B,'overall')*100:.1f}%")


if __name__ == "__main__":
    main()
