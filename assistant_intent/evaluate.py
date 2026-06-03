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
    threshold τ, answer `none`. We:
        1. show the top-similarity distributions (real vs none) — can τ separate them?
        2. sweep τ and print the real-acc / none-recall / overall-acc tradeoff
        3. pick the τ that maximizes overall 5-way accuracy and report it
    (Honesty caveat: τ is tuned on the same test set here for the demo; in a real
     pipeline you'd tune τ on a separate validation split.)

Run:  python evaluate.py            (after train.py has written models/finetuned/)
"""
import os
import sys
import argparse

import numpy as np
import torch

_CIS = os.path.join(os.path.dirname(__file__), "..", "customer_intent_search")
sys.path.insert(0, _CIS)
from tokenizer import SimpleTokenizer   # noqa: E402
from model import build_model           # noqa: E402

from shared import load_by_intent, build_prototypes   # noqa: E402  (shared helpers)

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
REAL_INTENTS = ["answers", "media", "smart_home", "productivity"]


# ── loading ──────────────────────────────────────────────────────────────────
def load_model(model_dir: str, device: str):
    """Rebuild the embedder from saved tokenizer + weights (architecture is
    derived from the tokenizer, exactly as in train.py)."""
    tok = SimpleTokenizer.load(os.path.join(model_dir, "tokenizer.json"))
    model = build_model(tok).to(device)
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, tok


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
        sims = emb @ protos.T                                    # [m,C]
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
        sims = emb @ protos.T
        out[it] = (sims.max(axis=1), sims.argmax(axis=1))
    return out


def evaluate_none(model, tok, test_groups, none_texts, protos, intents, device):
    real_top = top_sims(model, tok, test_groups, protos, device)
    none_emb = model.encode(none_texts, tok, device=device) if none_texts else np.empty((0, protos.shape[1]))
    none_sim = none_emb @ protos.T if len(none_emb) else np.empty((0, len(intents)))
    none_max = none_sim.max(axis=1) if len(none_sim) else np.empty(0)

    # flatten real-query top sims + whether argmax intent is correct
    real_max, real_correct = [], []
    for ti, it in enumerate(intents):
        mx, am = real_top[it]
        real_max.extend(mx.tolist())
        real_correct.extend((am == ti).tolist())
    real_max = np.array(real_max)
    real_correct = np.array(real_correct, dtype=bool)
    n_real, n_none = len(real_max), len(none_max)

    # threshold sweep: predict `none` when top sim < τ
    rows = []
    taus = np.round(np.arange(0.30, 0.96, 0.05), 2)
    for tau in taus:
        # a real query is right iff it clears τ AND its nearest prototype is correct
        real_right = int(np.sum(real_correct & (real_max >= tau)))
        none_right = int(np.sum(none_max < tau))     # correctly rejected
        overall = (real_right + none_right) / max(1, n_real + n_none)
        rows.append({
            "tau": float(tau),
            "real_acc": real_right / max(1, n_real),
            "none_recall": none_right / max(1, n_none),
            "overall": overall,
        })
    best = max(rows, key=lambda r: r["overall"])
    return real_max, none_max, rows, best, (n_real, n_none)


# ── reporting ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Evaluate the assistant intent embedder")
    ap.add_argument("--model-dir", default=os.path.join(HERE, "models", "finetuned"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tok = load_model(args.model_dir, device)
    train_groups = load_by_intent(os.path.join(DATA_DIR, "train.json"), REAL_INTENTS)
    test_groups = load_by_intent(os.path.join(DATA_DIR, "test.json"), REAL_INTENTS)
    none_texts = load_by_intent(os.path.join(DATA_DIR, "test.json"), ["none"])["none"]

    print("=" * 64)
    print("PART A — retrieval on the 4 REAL intents (nearest prototype)")
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
    print("PART B — the `none` problem and the threshold fix")
    print("=" * 64)
    real_max, none_max, rows, best, (n_real, n_none) = evaluate_none(
        model, tok, test_groups, none_texts, protos, intents, device)

    def pct(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")
    print(f"  top-similarity distribution (how close to the nearest prototype):")
    print(f"    {'group':<18} {'min':>6} {'25%':>6} {'median':>7} {'mean':>6} {'max':>6}   n")
    print(f"    {'real intents':<18} {real_max.min():6.2f} {pct(real_max,25):6.2f} "
          f"{np.median(real_max):7.2f} {real_max.mean():6.2f} {real_max.max():6.2f}  {n_real:>3}")
    print(f"    {'none (junk/OOS)':<18} {none_max.min():6.2f} {pct(none_max,25):6.2f} "
          f"{np.median(none_max):7.2f} {none_max.mean():6.2f} {none_max.max():6.2f}  {n_none:>3}")
    print("    ↑ if 'real' sits clearly above 'none', a threshold can separate them.")

    # without a threshold: none is ALWAYS assigned to some real intent → 0% recall
    print(f"\n  WITHOUT threshold: none recall = 0.0%  "
          f"(all {n_none} none queries forced into a real intent)")

    print(f"\n  threshold sweep (predict `none` when top similarity < τ):")
    print(f"    {'τ':>5} {'real_acc':>9} {'none_recall':>12} {'overall':>9}")
    for r in rows:
        mark = "  ← best" if r is best else ""
        print(f"    {r['tau']:>5.2f} {r['real_acc']*100:>8.1f}% "
              f"{r['none_recall']*100:>11.1f}% {r['overall']*100:>8.1f}%{mark}")

    print(f"\n  best τ = {best['tau']:.2f}: "
          f"real_acc {best['real_acc']*100:.1f}%, "
          f"none_recall {best['none_recall']*100:.1f}%, "
          f"overall 5-way {best['overall']*100:.1f}%")
    print(f"\n  → adding the threshold recovers `none` from 0.0% to "
          f"{best['none_recall']*100:.1f}% while keeping real-intent accuracy at "
          f"{best['real_acc']*100:.1f}%.")


if __name__ == "__main__":
    main()
