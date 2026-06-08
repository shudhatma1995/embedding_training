"""
eval_massive.py  -  the honest external-data verdict: ours vs MiniLM on MASSIVE.
================================================================================

Builds prototypes from OUR train.json, then classifies real Amazon MASSIVE
utterances (data/test_massive.json, produced by build_massive.py) through the SAME
evaluate_model harness used for the template and wild sets. Runs our canonical
model and zero-shot MiniLM side by side so the only variable is the encoder.

This is the test our own synthetic/wild sets couldn't be — third-party data we did
not author. Expect the numbers to be lower than template/wild and the gap to MiniLM
(pretrained on real language) to widen: that is the point, not a bug.

Run:  python eval_massive.py     (needs data/test_massive.json; run build_massive.py first)
"""

import os

import torch
from baseline_minilm import make_minilm_encoder
from data import load_eval_set
from evaluate import evaluate_model, load_model
from intents import REAL_INTENTS
from shared import make_encoder

HERE = os.path.dirname(__file__)


def _row(name, res):
    h = res["honest"]
    return (
        f"  {name:<22} {res['micro']['recall@1'] * 100:6.1f}% {res['micro']['mrr']:6.3f} "
        f"{res['real_sim']:7.2f} {res['none_sim']:7.2f}   "
        f"{h['overall'] * 100:6.1f}% {h['none_recall'] * 100:7.1f}%"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # our model dictates the intent set (match what it was trained on)
    ours_model, ours_tok, config = load_model(os.path.join(HERE, "models", "finetuned"), device)
    real_intents = config.get("train_intents") or REAL_INTENTS
    train_groups, massive_groups, massive_none = load_eval_set("massive", real_intents)

    n_real = sum(len(v) for v in massive_groups.values())
    print(f"  Intents ({len(real_intents)}): {', '.join(real_intents)}")
    print(f"  MASSIVE queries: {n_real} real + {len(massive_none)} none")
    print("  Loading MiniLM baseline ...")
    _, mini_encode = make_minilm_encoder(device=device)
    ours_encode = make_encoder(ours_model, ours_tok, device)

    ours = evaluate_model(ours_encode, train_groups, massive_groups, massive_none)
    mini = evaluate_model(mini_encode, train_groups, massive_groups, massive_none)

    print("\n" + "=" * 78)
    print("  EXTERNAL DATA (Amazon MASSIVE) — ours vs all-MiniLM-L6-v2, same harness")
    print("=" * 78)
    print(
        f"  {'model':<22} {'R@1':>7} {'MRR':>6} {'realSim':>7} {'noneSim':>7}   "
        f"{'honest':>7} {'noneRec':>8}"
    )
    print(_row("ours (from scratch)", ours))
    print(_row("MiniLM (zero-shot)", mini))

    print("\n  per-intent R@1 (ours | MiniLM):")
    print(f"    {'intent':<14} {'ours':>7} {'MiniLM':>8}")
    for it in ours["intents"]:
        print(
            f"    {it:<14} {ours['per_intent'][it]['recall@1'] * 100:6.1f}% "
            f"{mini['per_intent'][it]['recall@1'] * 100:7.1f}%"
        )

    print(
        "\n  read: this is OUT-OF-DISTRIBUTION third-party data. Compare R@1 here to "
        "the template/wild\n  numbers — the drop is the real generalization gap."
    )


if __name__ == "__main__":
    main()
