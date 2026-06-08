"""
hn_ablation.py  -  does per-anchor hard-negative mining actually help? (honestly)
================================================================================

Why this exists
  Hard-negative mining sharpens confusable boundaries (media vs smart_home) by
  feeding each anchor its OWN most-confusable different-intent query as an extra
  loss column (train.py mnr_loss / shared.mine_hard_negatives). The question is
  whether that helps THIS model — and the project's hard-won rule is: never trust
  a single-seed delta on a tiny model. So we sweep hn_top_k over several seeds and
  report mean ± std.

What it measures (two test sets, because they disagree)
  TEMPLATE  data/test.json       — in-distribution (same generator as train)
  WILD      data/test_wild.json  — hand-written OOD phrasing; the honest gap.
  Hard negatives target confusable boundaries, which is exactly the OOD failure
  mode, so WILD is where a real win should show (or fail to).

Everything else is held at the canonical config B (none-in-vocab + none-neg-k=8),
so the ONLY thing changing across rows is hn_top_k. k=0 is the no-HN baseline.

Run:  python hn_ablation.py                  (k in {0,1,3,5}, 5 seeds, ~5 min CPU)
      python hn_ablation.py --seeds 0 1 2     (fewer seeds)
      python hn_ablation.py --top-ks 0 3      (fewer configs)
"""

import argparse
import contextlib
import importlib.util
import io
import os
import shutil

import numpy as np
import torch
from data import load_eval_set
from intents import REAL_INTENTS

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_local(modname, filename):
    """Load a sibling module by explicit path (same shim as experiments.py — keeps
    customer_intent_search/evaluate.py on sys.path from shadowing our evaluate.py)."""
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load_local("ai_train", "train.py")
ev = _load_local("ai_evaluate", "evaluate.py")

# (metric key, header, percent?) — drives the table columns.
METRIC_COLS = [
    ("tmpl_r1", "tmpl R@1", True),
    ("tmpl_overall", "tmpl overall", True),
    ("wild_r1", "wild R@1", True),
    ("wild_overall", "wild overall", True),
]


def run_one(top_k, seed, device, tmpl, wild, scratch):
    """Train config B at one hn_top_k + seed (silently), eval on template AND wild."""
    args = tr.build_parser().parse_args([])  # inherit train.py default hyperparams
    args.seed = seed
    args.none_in_vocab = True  # canonical config B
    args.none_neg_k = 8
    args.hn_top_k = top_k  # the only knob we vary
    args.output_dir = scratch
    with contextlib.redirect_stdout(io.StringIO()):  # silence the epoch table
        model, tok = tr.train(args)

    encode = ev.make_encoder(model, tok, device)
    rt = ev.evaluate_model(encode, *tmpl)
    rw = ev.evaluate_model(encode, *wild)
    return {
        "tmpl_r1": rt["micro"]["recall@1"],
        "tmpl_overall": rt["test_tuned"]["overall"],
        "wild_r1": rw["micro"]["recall@1"],
        "wild_overall": rw["test_tuned"]["overall"],
    }


def ms(vals):
    """'mean±std' as percentages."""
    a = np.array(vals)
    return f"{a.mean() * 100:4.1f}±{a.std() * 100:3.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--top-ks", type=int, nargs="+", default=[0, 1, 3, 5])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scratch = os.path.join(HERE, "models", "_hn_ablation_tmp")

    tmpl = load_eval_set("template", REAL_INTENTS)  # (train, template-test, template-none)
    wild = load_eval_set("wild", REAL_INTENTS)  # (train, wild-test, wild-none)

    print("=" * 78)
    print(f"HARD-NEGATIVE ABLATION  (config B; k in {args.top_ks}; seeds={args.seeds})")
    print("=" * 78)

    results = {k: [] for k in args.top_ks}
    for seed in args.seeds:
        for k in args.top_ks:
            m = run_one(k, seed, device, tmpl, wild, scratch)
            results[k].append(m)
            print(
                f"  seed {seed} | hn_top_k={k} | "
                f"tmpl R@1 {m['tmpl_r1'] * 100:5.1f}%  wild R@1 {m['wild_r1'] * 100:5.1f}%  "
                f"wild overall {m['wild_overall'] * 100:5.1f}%"
            )
    shutil.rmtree(scratch, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"  MEAN ± STD over {len(args.seeds)} seeds")
    print("=" * 78)
    print(f"  {'hn_top_k':<10}" + "".join(f"{h:>16}" for _, h, _ in METRIC_COLS))
    for k in args.top_ks:
        row = f"  {k:<10}"
        for key, _, _ in METRIC_COLS:
            row += f"{ms([r[key] for r in results[k]]):>16}"
        print(row)

    # PAIRED deltas vs k=0 — the statistically right test here. For a given seed,
    # k=0 and k>0 share the same init AND warmup (bit-identical until mining fires),
    # so they are PAIRED runs. Comparing marginal ±std bands throws that pairing away
    # and is far too conservative; the per-seed delta cancels the init noise that
    # dominates the marginal spread. (Extends the project's seed-averaging lesson:
    # average over seeds, but when runs are paired, difference WITHIN each seed.)
    if 0 in results and len(args.top_ks) > 1:
        print("\n  PAIRED Δ vs k=0  (per-seed difference cancels init noise):")
        print(f"  {'metric':<10}" + "".join(f"  k={k:<11}" for k in args.top_ks if k != 0))
        for key, hdr, _ in METRIC_COLS:
            row = f"  {hdr:<10}"
            for k in args.top_ks:
                if k == 0:
                    continue
                d = np.array(
                    [results[k][i][key] - results[0][i][key] for i in range(len(args.seeds))]
                )
                wins = int((d > 0).sum())
                row += f"  {d.mean() * 100:+4.1f}±{d.std() * 100:3.1f} {wins}/{len(d)}"
            print(row)
        print(
            "  (Δ ± std of the per-seed deltas, and #seeds improved. A delta whose "
            "sign is consistent\n   across seeds is real even if the marginal ±std bands above overlap.)"
        )


if __name__ == "__main__":
    main()
