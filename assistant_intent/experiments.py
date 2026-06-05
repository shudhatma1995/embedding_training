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

# Each config: a short stable `id` (referenced by the verdicts) + a human `label`.
# Keying results by id means renaming a label can't silently break the verdicts.
CONFIGS = [
    {"id": "A", "label": "real-only-vocab", "none_in_vocab": False, "none_neg_k": 0},
    {"id": "C", "label": "none-in-vocab",   "none_in_vocab": True,  "none_neg_k": 0},
    {"id": "B", "label": "+negatives k=8",  "none_in_vocab": True,  "none_neg_k": 8},
]

# (metric key, column header, format-as-percent?) — one place drives table + per-seed line.
METRIC_COLS = [
    ("r1", "real R@1", True), ("real_sim", "real sim", False),
    ("none_sim", "none sim", False), ("overall", "best overall", True),
    ("none_recall", "none recall", True),
]


def display(cfg):
    """Human label for a config row, e.g. 'A real-only-vocab'."""
    return f"{cfg['id']} {cfg['label']}"


def run_one(cfg, seed, device, train_groups, test_groups, none_texts, scratch):
    """Train one config at one seed (output suppressed) and return eval metrics."""
    # Inherit train.py's default hyperparameters from its own parser, so they can't
    # silently drift; override only the ablation knobs (intents stays None = full set).
    args = tr.build_parser().parse_args([])
    args.seed = seed
    args.none_neg_k = cfg["none_neg_k"]
    args.none_in_vocab = cfg["none_in_vocab"]
    args.output_dir = scratch
    with contextlib.redirect_stdout(io.StringIO()):     # silence the epoch table
        model, tok = tr.train(args)

    encode = ev.make_encoder(model, tok, device)
    res = ev.evaluate_model(encode, train_groups, test_groups, none_texts)
    best = res["test_tuned"]            # full-test τ-optimum (relative config comparison)
    return {
        "r1": res["micro"]["recall@1"],
        "real_sim": res["real_sim"],
        "none_sim": res["none_sim"],
        "overall": best["overall"],
        "none_recall": best["none_recall"],
    }


def ms(vals, pct=False):
    """Format a list as 'mean±std' (as percentages if pct)."""
    a = np.array(vals)
    if pct:
        return f"{a.mean()*100:4.1f}±{a.std()*100:3.1f}"
    return f"{a.mean():.2f}±{a.std():.2f}"


def run_grid(seeds, device, data, scratch):
    """Train every (config × seed) and collect metrics → {config_id: [per-seed dicts]}."""
    train_groups, test_groups, none_texts = data
    results = {c["id"]: [] for c in CONFIGS}
    for seed in seeds:
        for cfg in CONFIGS:
            m = run_one(cfg, seed, device, train_groups, test_groups, none_texts, scratch)
            results[cfg["id"]].append(m)
            print(f"  seed {seed} | {display(cfg):<20} "
                  f"R@1 {m['r1']*100:5.1f}%  none_sim {m['none_sim']:.2f}  "
                  f"overall {m['overall']*100:5.1f}%")
    return results


def print_table(results, n_seeds):
    """The mean±std summary table, one row per config (in CONFIGS order)."""
    print("\n" + "=" * 78)
    print(f"  MEAN ± STD over {n_seeds} seeds")
    print("=" * 78)
    print(f"  {'config':<20}" + "".join(f"{lbl:>14}" for _, lbl, _ in METRIC_COLS))
    for cfg in CONFIGS:
        runs = results[cfg["id"]]
        row = f"  {display(cfg):<20}"
        for key, _, pct in METRIC_COLS:
            row += f"{ms([r[key] for r in runs], pct):>14}"
        print(row)


def print_verdicts(results):
    """The two clean comparisons, referencing configs by id (A/B/C), not by label."""
    def mean(cid, key):
        return float(np.mean([r[key] for r in results[cid]]))

    def verdict(title, left, right):
        print(f"    {title:<26}: real R@1 {mean(left,'r1')*100:.1f}% → {mean(right,'r1')*100:.1f}%, "
              f"none_sim {mean(left,'none_sim'):.2f} → {mean(right,'none_sim'):.2f}, "
              f"overall {mean(left,'overall')*100:.1f}% → {mean(right,'overall')*100:.1f}%")

    print("\n  verdicts:")
    verdict("A vs C (vocab effect)", "A", "C")
    verdict("C vs B (negatives effect)", "C", "B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scratch = os.path.join(HERE, "models", "_ablation_tmp")

    data = (
        ev.load_by_intent(os.path.join(ev.DATA_DIR, "train.json"), ev.REAL_INTENTS),
        ev.load_by_intent(os.path.join(ev.DATA_DIR, "test.json"), ev.REAL_INTENTS),
        ev.load_by_intent(os.path.join(ev.DATA_DIR, "test.json"), [ev.NONE_ID])[ev.NONE_ID],
    )

    print("=" * 78)
    print(f"MULTI-SEED ABLATION  (seeds={args.seeds}, {len(args.seeds)} runs per config)")
    print("=" * 78)

    results = run_grid(args.seeds, device, data, scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    print_table(results, len(args.seeds))
    print_verdicts(results)


if __name__ == "__main__":
    main()
