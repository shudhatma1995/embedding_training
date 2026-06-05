"""
evaluate.py  -  Stage 5: measure the trained intent classifier honestly.
================================================================================

Two parts, matching the two things we care about:

PART A  -  Retrieval quality on the REAL intents
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
import json
import random
import argparse

import numpy as np
import torch

# Embedder + tokenizer come from customer_intent_search (reused untouched) via cis.py.
from cis import SimpleTokenizer, build_model
from data import load_eval_data        # standard (train, test, none) loader
from shared import build_prototypes, safe_matmul, make_encoder   # encoder-agnostic kernel
from intents import REAL_INTENTS       # taxonomy source of truth

HERE = os.path.dirname(__file__)


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


# build_prototypes lives in shared.py (imported above) and takes an Encoder.


# ── PART A: retrieval metrics (given prototypes) ─────────────────────────────
def retrieval_metrics(encode, test_groups, protos, intents):
    """Recall@1 / MRR (micro + per-intent) + the misclassified list, over the real
    intents. `encode` is the Encoder; `protos`/`intents` come from build_prototypes."""
    per_intent = {}
    rr_all, hit1_all = [], []
    misclassified = []   # (text, true, predicted, top_sim)

    for ti, it in enumerate(intents):
        texts = test_groups[it]
        if not texts:
            continue
        emb = encode(texts)                                      # [m,D]
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
    return micro, per_intent, misclassified


# ── PART B: none handling via similarity threshold ───────────────────────────
def top_sims(encode, groups, protos):
    """For each query return its TOP similarity to any prototype (+ argmax intent)."""
    out = {}
    for it, texts in groups.items():
        if not texts:
            out[it] = (np.empty(0), np.empty(0, dtype=int))
            continue
        emb = encode(texts)
        sims = safe_matmul(emb, protos.T)
        out[it] = (sims.max(axis=1), sims.argmax(axis=1))
    return out


TAUS = np.round(np.arange(0.30, 0.96, 0.05), 2)


def query_sims(encode, groups, none_texts, protos, intents):
    """Per-query TOP similarity to any prototype, for real + none queries.
    Returns (real_max [N], real_correct [N] bool, none_max [M])."""
    real_top = top_sims(encode, groups, protos)
    none_emb = encode(none_texts) if none_texts else np.empty((0, protos.shape[1]))
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


def evaluate_model(encode, train_groups, test_groups, none_texts, val_frac=0.5, split_seed=0):
    """ONE evaluation, shared by evaluate.py, experiments.py, and the MiniLM baseline.
    Builds prototypes from `train_groups` using `encode`, then computes everything
    any caller needs (each reads the fields it wants):

      PART A      micro/per-intent Recall@1 + MRR, misclassified list
      sims        real_sim / none_sim means + the full-test τ-optimum (`test_tuned`)
      honest      τ TUNED on a val half, REPORTED on the untouched test half
                  (+ `optimistic_split`, the test-half optimum, to show the bias)

    Because it depends only on `encode`, the same call evaluates our from-scratch
    model and a pretrained baseline identically — an apples-to-apples comparison.
    """
    protos, intents = build_prototypes(encode, train_groups)
    micro, per_intent, misclassified = retrieval_metrics(encode, test_groups, protos, intents)

    # full-test none-threshold sims → real_sim/none_sim + the (optimistic) test optimum
    real_max, real_correct, none_max = query_sims(encode, test_groups, none_texts, protos, intents)
    test_tuned = pick_best(tau_sweep(real_max, real_correct, none_max))

    # honest τ: tune on a VAL half, freeze, report on the untouched TEST half
    val_g, val_none, te_g, te_none = split_val_test(test_groups, none_texts, val_frac, split_seed)
    rmv, rcv, nmv = query_sims(encode, val_g, val_none, protos, intents)
    rmt, rct, nmt = query_sims(encode, te_g, te_none, protos, intents)
    val_rows = tau_sweep(rmv, rcv, nmv)
    tau = pick_best(val_rows)["tau"]
    honest = metrics_at(tau, rmt, rct, nmt)
    optimistic = pick_best(tau_sweep(rmt, rct, nmt))

    return {
        "protos": protos, "intents": intents,
        "micro": micro, "per_intent": per_intent, "misclassified": misclassified,
        "real_sim": float(real_max.mean()), "none_sim": float(none_max.mean()),
        "test_tuned": test_tuned,                    # full-test optimum (relative/ablation)
        "honest": honest, "honest_tau": tau,         # val-tuned → test-reported (headline)
        "optimistic_split": optimistic,              # test-half optimum (bias illustration)
        "val_rows": val_rows,                        # for the printed val sweep
        "test_real_max": rmt, "test_none_max": nmt,  # for the distribution table
    }


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
    train_groups, test_groups, none_texts = load_eval_data(real_intents)

    encode = make_encoder(model, tok, device)
    res = evaluate_model(encode, train_groups, test_groups, none_texts)
    intents = res["intents"]

    print("=" * 64)
    print("PART A — retrieval on the REAL intents (nearest prototype)")
    print("=" * 64)
    print(f"  micro   Recall@1 {res['micro']['recall@1']*100:5.1f}%   MRR {res['micro']['mrr']:.3f}")
    print(f"  {'intent':<13} {'R@1':>7} {'MRR':>7} {'n':>4}")
    for it in intents:
        m = res["per_intent"][it]
        print(f"  {it:<13} {m['recall@1']*100:6.1f}% {m['mrr']:7.3f} {m['n']:>4}")

    misclassified = res["misclassified"]
    print(f"\n  misclassified ({len(misclassified)}):")
    if not misclassified:
        print("    (none)")
    for t, true, pred, s in misclassified:
        print(f"    [{true:>12} → {pred:<12}] sim={s:.2f}  \"{t}\"")

    print("\n" + "=" * 64)
    print("PART B — the `none` problem and an HONEST threshold (τ tuned on VAL)")
    print("=" * 64)

    rmax_t, nmax_t = res["test_real_max"], res["test_none_max"]

    def pct(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")
    print(f"  top-similarity distribution on the TEST half (real vs none):")
    print(f"    {'group':<18} {'min':>6} {'25%':>6} {'median':>7} {'mean':>6} {'max':>6}   n")
    print(f"    {'real intents':<18} {rmax_t.min():6.2f} {pct(rmax_t,25):6.2f} "
          f"{np.median(rmax_t):7.2f} {rmax_t.mean():6.2f} {rmax_t.max():6.2f}  {len(rmax_t):>3}")
    print(f"    {'none (junk/OOS)':<18} {nmax_t.min():6.2f} {pct(nmax_t,25):6.2f} "
          f"{np.median(nmax_t):7.2f} {nmax_t.mean():6.2f} {nmax_t.max():6.2f}  {len(nmax_t):>3}")

    tau = res["honest_tau"]
    print(f"\n  τ sweep on the VAL half (we pick τ here, then FREEZE it):")
    print(f"    {'τ':>5} {'real_acc':>9} {'none_recall':>12} {'overall':>9}")
    for r in res["val_rows"]:
        mark = "  ← picked" if r["tau"] == tau else ""
        print(f"    {r['tau']:>5.2f} {r['real_acc']*100:>8.1f}% "
              f"{r['none_recall']*100:>11.1f}% {r['overall']*100:>8.1f}%{mark}")

    honest, opt = res["honest"], res["optimistic_split"]
    print(f"\n  HONEST     (τ={tau:.2f} tuned on VAL → reported on TEST):")
    print(f"    real_acc {honest['real_acc']*100:.1f}%, none_recall {honest['none_recall']*100:.1f}%, "
          f"overall {honest['overall']*100:.1f}%")
    print(f"  OPTIMISTIC (τ={opt['tau']:.2f} tuned ON test — the old, biased way):")
    print(f"    overall {opt['overall']*100:.1f}%")
    print(f"\n  → optimism bias = {(opt['overall']-honest['overall'])*100:+.1f} points. "
          f"Tuning τ on the test set overstates accuracy; the val-tuned number is the honest one.")
    print(f"  (small-n caveat: each half is ~{len(rmax_t)} real + {len(nmax_t)} none queries, so "
          f"these move seed-to-seed — the methodology is the point, not the exact figure.)")


if __name__ == "__main__":
    main()
