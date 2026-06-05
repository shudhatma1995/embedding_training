"""
evaluate.py  -  Stage 5: measure the trained intent classifier honestly.
================================================================================

Two parts, matching the two things we care about:

PART A  -  Retrieval quality on the 4 REAL intents
    The classifier is "nearest prototype": prototype[intent] = centroid (mean) of
    that intent's TRAIN embeddings. A test query is labelled by the most similar
    prototype. We report:
        Recall@1  - was the correct intent the single nearest prototype?
        MRR       - 1/rank of the correct intent, averaged (rewards near-misses)
    micro (over all queries) + per-intent, and we PRINT the actual queries it gets
    wrong — those are the boundary cases worth studying.

PART B  -  The `none` problem and the threshold fix
    A nearest-prototype classifier ALWAYS returns one of the known intents. So an
    out-of-scope query ("asdfgh", "book me a flight") is confidently mislabelled —
    none recall is 0%. The fix from intents.py: if the TOP similarity is below a
    threshold τ, answer `none`. To pick τ HONESTLY we split the eval set in half:
        1. show the top-similarity distributions (real vs none) — can τ separate them?
        2. TUNE τ on the VAL half (sweep, pick the τ that maximizes overall accuracy)
        3. FREEZE τ and REPORT it on the untouched TEST half
    We also print the OPTIMISTIC number (τ tuned directly on test) to show the
    optimism bias — choosing a hyperparameter on the eval set overstates accuracy.
    (Caveat: τ used to be tuned on the test set itself; that was the honesty hole
     this split closes. The halves are small, so the exact figures move by seed.)

Run:  python evaluate.py            (after train.py has written models/finetuned/)
"""
import os
import sys
import json
import random
import argparse

import numpy as np
import torch

_CIS = os.path.join(os.path.dirname(__file__), "..", "customer_intent_search")
sys.path.insert(0, _CIS)
from tokenizer import SimpleTokenizer   # noqa: E402
from model import build_model           # noqa: E402

from data import load_by_intent       # noqa: E402  (json I/O lives in data.py)
from shared import build_prototypes, safe_matmul   # noqa: E402  (kernel + FPE-safe matmul)
from intents import REAL_INTENTS, NONE_ID   # noqa: E402  (taxonomy source of truth)

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")


# ── loading ──────────────────────────────────────────────────────────────────
def load_model(model_dir: str, device: str):
    """Rebuild the embedder from saved tokenizer + weights (architecture is
    derived from the tokenizer, exactly as in train.py). Also returns the saved
    config.json so eval can mirror the EXACT intents the model was trained on."""
    tok = SimpleTokenizer.load(os.path.join(model_dir, "tokenizer.json"))
    model = build_model(tok).to(device)
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    config_path = os.path.join(model_dir, "config.json")
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    return model, tok, config


# ── prototypes ───────────────────────────────────────────────────────────────
# build_prototypes now lives in protos.py (shared with train.py) — imported above.


# ── PART A: retrieval metrics ────────────────────────────────────────────────
def evaluate_real(model, tok, train_groups, test_groups, device):
    protos, intents = build_prototypes(model, tok, train_groups, device)

    per_intent = {}
    rr_all, hit1_all = [], []
    misclassified = []   # (text, true, predicted, top_sim)

    for ti, it in enumerate(intents):
        texts = test_groups[it]
        if not texts:
            continue
        emb = model.encode(texts, tok, device=device)            # [m,D]
        sims = safe_matmul(emb, protos.T)                        # [m,C]
        ranked = np.argsort(-sims, axis=1)                       # [m,C], best first

        hit1 = (ranked[:, 0] == ti).astype(float)
        # rank (1-based) of the correct intent for each query → reciprocal rank
        rank_of_true = np.argmax(ranked == ti, axis=1) + 1
        rr = 1.0 / rank_of_true

        per_intent[it] = {"recall@1": hit1.mean(), "mrr": rr.mean(), "n": len(texts)}
        hit1_all.extend(hit1.tolist())
        rr_all.extend(rr.tolist())

        for j, t in enumerate(texts):
            if ranked[j, 0] != ti:
                misclassified.append((t, it, intents[ranked[j, 0]], float(sims[j].max())))

    micro = {"recall@1": float(np.mean(hit1_all)), "mrr": float(np.mean(rr_all))}
    return micro, per_intent, misclassified, (protos, intents)


# ── PART B: none handling via similarity threshold ───────────────────────────
def top_sims(model, tok, groups, protos, device):
    """For each query return its TOP similarity to any prototype (+ argmax intent)."""
    out = {}
    for it, texts in groups.items():
        if not texts:
            out[it] = (np.empty(0), np.empty(0, dtype=int))
            continue
        emb = model.encode(texts, tok, device=device)
        sims = safe_matmul(emb, protos.T)
        out[it] = (sims.max(axis=1), sims.argmax(axis=1))
    return out


TAUS = np.round(np.arange(0.30, 0.96, 0.05), 2)


def query_sims(model, tok, groups, none_texts, protos, intents, device):
    """Per-query TOP similarity to any prototype, for real + none queries.
    Returns (real_max [N], real_correct [N] bool, none_max [M])."""
    real_top = top_sims(model, tok, groups, protos, device)
    none_emb = model.encode(none_texts, tok, device=device) if none_texts else np.empty((0, protos.shape[1]))
    none_sim = safe_matmul(none_emb, protos.T) if len(none_emb) else np.empty((0, len(intents)))
    none_max = none_sim.max(axis=1) if len(none_sim) else np.empty(0)

    real_max, real_correct = [], []
    for ti, it in enumerate(intents):
        mx, am = real_top[it]
        real_max.extend(mx.tolist())
        real_correct.extend((am == ti).tolist())
    return np.array(real_max), np.array(real_correct, dtype=bool), none_max


def metrics_at(tau, real_max, real_correct, none_max):
    """Accuracy if we answer `none` when top sim < τ. A real query is right iff it
    clears τ AND its nearest prototype is the correct intent."""
    n_real, n_none = len(real_max), len(none_max)
    real_right = int(np.sum(real_correct & (real_max >= tau)))
    none_right = int(np.sum(none_max < tau))          # correctly rejected
    return {"tau": float(tau),
            "real_acc": real_right / max(1, n_real),
            "none_recall": none_right / max(1, n_none),
            "overall": (real_right + none_right) / max(1, n_real + n_none)}


def tau_sweep(real_max, real_correct, none_max, taus=TAUS):
    return [metrics_at(t, real_max, real_correct, none_max) for t in taus]


def pick_best(rows):
    return max(rows, key=lambda r: r["overall"])


def evaluate_none(model, tok, test_groups, none_texts, protos, intents, device):
    """Sweep τ on the GIVEN set and return the best by overall accuracy.
    Used by experiments.py for relative config comparison (τ tuned on the set passed
    in). For an HONEST single-model number, main() tunes τ on a separate val split."""
    real_max, real_correct, none_max = query_sims(
        model, tok, test_groups, none_texts, protos, intents, device)
    rows = tau_sweep(real_max, real_correct, none_max)
    return real_max, none_max, rows, pick_best(rows), (len(real_max), len(none_max))


def split_val_test(groups, none_texts, frac=0.5, seed=0):
    """Stratified split of the eval queries into a VAL half (to TUNE τ) and a TEST
    half (to REPORT). Per-intent lists and the none list are split independently so
    both halves keep every class. Seeded for reproducibility."""
    rng = random.Random(seed)

    def split(items):
        items = list(items)
        rng.shuffle(items)
        k = round(len(items) * frac)
        return items[:k], items[k:]               # (val, test)

    val_g, test_g = {}, {}
    for it, texts in groups.items():
        val_g[it], test_g[it] = split(texts)
    val_none, test_none = split(none_texts)
    return val_g, val_none, test_g, test_none


# ── reporting ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluate the assistant intent embedder")
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models", "finetuned"))
    ap.add_argument("--intents", nargs="+", default=None,
                    help="override the intent set to evaluate (default: the model's "
                         "trained intents from config.json).")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tok, config = load_model(args.model_dir, device)
    # Follow the model by default (no train/eval mismatch possible); --intents overrides;
    # REAL_INTENTS is the last-resort fallback for a config without train_intents.
    real_intents = args.intents or config.get("train_intents") or REAL_INTENTS
    print(f"  Evaluating intents: {', '.join(real_intents)}\n")
    train_groups = load_by_intent(os.path.join(DATA_DIR, "train.json"), real_intents)
    test_groups = load_by_intent(os.path.join(DATA_DIR, "test.json"), real_intents)
    none_texts = load_by_intent(os.path.join(DATA_DIR, "test.json"), [NONE_ID])[NONE_ID]

    print("=" * 64)
    print("PART A — retrieval on the REAL intents (nearest prototype)")
    print("=" * 64)
    micro, per_intent, misclassified, (protos, intents) = evaluate_real(
        model, tok, train_groups, test_groups, device)
    print(f"  micro   Recall@1 {micro['recall@1']*100:5.1f}%   MRR {micro['mrr']:.3f}")
    print(f"  {'intent':<13} {'R@1':>7} {'MRR':>7} {'n':>4}")
    for it in intents:
        m = per_intent[it]
        print(f"  {it:<13} {m['recall@1']*100:6.1f}% {m['mrr']:7.3f} {m['n']:>4}")

    print(f"\n  misclassified ({len(misclassified)}):")
    if not misclassified:
        print("    (none)")
    for t, true, pred, s in misclassified:
        print(f"    [{true:>12} → {pred:<12}] sim={s:.2f}  \"{t}\"")

    print("\n" + "=" * 64)
    print("PART B — the `none` problem and an HONEST threshold (τ tuned on VAL)")
    print("=" * 64)

    # Split the eval set: TUNE τ on the val half, REPORT on the untouched test half.
    val_g, val_none, te_g, te_none = split_val_test(test_groups, none_texts, frac=0.5, seed=0)
    rmax_v, rcorr_v, nmax_v = query_sims(model, tok, val_g, val_none, protos, intents, device)
    rmax_t, rcorr_t, nmax_t = query_sims(model, tok, te_g, te_none, protos, intents, device)

    def pct(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")
    print(f"  top-similarity distribution on the TEST half (real vs none):")
    print(f"    {'group':<18} {'min':>6} {'25%':>6} {'median':>7} {'mean':>6} {'max':>6}   n")
    print(f"    {'real intents':<18} {rmax_t.min():6.2f} {pct(rmax_t,25):6.2f} "
          f"{np.median(rmax_t):7.2f} {rmax_t.mean():6.2f} {rmax_t.max():6.2f}  {len(rmax_t):>3}")
    print(f"    {'none (junk/OOS)':<18} {nmax_t.min():6.2f} {pct(nmax_t,25):6.2f} "
          f"{np.median(nmax_t):7.2f} {nmax_t.mean():6.2f} {nmax_t.max():6.2f}  {len(nmax_t):>3}")

    # tune τ on the VAL half (the test half is never used for selection)
    rows_val = tau_sweep(rmax_v, rcorr_v, nmax_v)
    best_val = pick_best(rows_val)
    tau = best_val["tau"]
    print(f"\n  τ sweep on the VAL half (we pick τ here, then FREEZE it):")
    print(f"    {'τ':>5} {'real_acc':>9} {'none_recall':>12} {'overall':>9}")
    for r in rows_val:
        mark = "  ← picked" if r is best_val else ""
        print(f"    {r['tau']:>5.2f} {r['real_acc']*100:>8.1f}% "
              f"{r['none_recall']*100:>11.1f}% {r['overall']*100:>8.1f}%{mark}")

    # HONEST: the frozen τ applied to the untouched TEST half
    honest = metrics_at(tau, rmax_t, rcorr_t, nmax_t)
    # OPTIMISTIC (biased): the BEST τ chosen ON the test half — what we'd report if we cheated
    best_test = pick_best(tau_sweep(rmax_t, rcorr_t, nmax_t))

    print(f"\n  HONEST     (τ={tau:.2f} tuned on VAL → reported on TEST):")
    print(f"    real_acc {honest['real_acc']*100:.1f}%, none_recall {honest['none_recall']*100:.1f}%, "
          f"overall {honest['overall']*100:.1f}%")
    print(f"  OPTIMISTIC (τ={best_test['tau']:.2f} tuned ON test — the old, biased way):")
    print(f"    overall {best_test['overall']*100:.1f}%")
    print(f"\n  → optimism bias = {(best_test['overall']-honest['overall'])*100:+.1f} points. "
          f"Tuning τ on the test set overstates accuracy; the val-tuned number is the honest one.")
    print(f"  (small-n caveat: each half is ~{len(rmax_t)} real + {len(nmax_t)} none queries, so "
          f"these move seed-to-seed — the methodology is the point, not the exact figure.)")


if __name__ == "__main__":
    main()
