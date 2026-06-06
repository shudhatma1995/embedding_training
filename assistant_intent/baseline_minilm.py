"""
baseline_minilm.py  -  Thread 3: calibrate our from-scratch model vs pretrained MiniLM.
================================================================================
"Is 95% R@1 good?" only means something against a yardstick. all-MiniLM-L6-v2 is
the standard one: ~22M params pretrained on >1B sentence pairs (vs our ~0.46M
params trained on ~800 generated sentences). We run BOTH through the SAME harness
(shared.evaluate_model) so the only difference is the encoder — an apples-to-apples
comparison.

The encoder abstraction is what makes this trivial: MiniLM just needs its own
`encode(texts) -> [N, D] L2-normalized` adapter. We mean-pool the token embeddings
(mask-aware) — the documented pooling for all-MiniLM-L6-v2 — then L2-normalize.

Requires:  pip install transformers   (the venv is otherwise torch + numpy only)
Run:       python baseline_minilm.py  (first run downloads ~90MB to ~/.cache/huggingface)
"""
import os

import numpy as np
import torch

# Local modules first (none of these put customer_intent_search/ on sys.path except
# `evaluate`, via cis — and by then `evaluate` has already resolved to ours).
from data import load_eval_data
from intents import REAL_INTENTS
from shared import make_encoder
from evaluate import evaluate_model, load_model

from transformers import AutoTokenizer, AutoModel   # noqa: E402

HERE = os.path.dirname(__file__)
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def make_minilm_encoder(model_name=MINILM, device="cpu", batch_size=64):
    """Load a pretrained HF model and return an Encoder: texts -> [N, D] L2-normed.
    Mean-pools the last hidden state over real (non-pad) tokens, then normalizes —
    the recommended pooling for all-MiniLM-L6-v2."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode(texts):
        if not texts:
            return np.empty((0, model.config.hidden_size), dtype=np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                          max_length=128, return_tensors="pt").to(device)
                hidden = model(**enc).last_hidden_state              # [B, L, D]
                mask = enc["attention_mask"].unsqueeze(-1).float()   # [B, L, 1]
                summed = (hidden * mask).sum(dim=1)                  # [B, D]
                counts = mask.sum(dim=1).clamp(min=1e-9)             # [B, 1]
                mean = summed / counts                              # mean over real tokens
                mean = torch.nn.functional.normalize(mean, p=2, dim=1)
                out.append(mean.cpu().numpy())
        return np.concatenate(out, axis=0)

    return model, encode   # tok is captured in the closure; no need to return it


def _row(name, res):
    h = res["honest"]
    return (f"  {name:<22} {res['micro']['recall@1']*100:6.1f}% {res['micro']['mrr']:6.3f} "
            f"{res['real_sim']:7.2f} {res['none_sim']:7.2f}   "
            f"{h['overall']*100:6.1f}% {h['none_recall']*100:7.1f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # our model decides the intent set (mirror exactly what it was trained on)
    ours_model, ours_tok, config = load_model(os.path.join(HERE, "models", "finetuned"), device)
    real_intents = config.get("train_intents") or REAL_INTENTS
    train_groups, test_groups, none_texts = load_eval_data(real_intents)
    print(f"  Intents ({len(real_intents)}): {', '.join(real_intents)}")
    print(f"  Loading {MINILM} ...")
    mini_model, mini_encode = make_minilm_encoder(device=device)

    ours_encode = make_encoder(ours_model, ours_tok, device)

    ours = evaluate_model(ours_encode, train_groups, test_groups, none_texts)
    mini = evaluate_model(mini_encode, train_groups, test_groups, none_texts)

    ours_params = ours_model.n_parameters()
    mini_params = sum(p.numel() for p in mini_model.parameters())

    print("\n" + "=" * 78)
    print("  OURS (from scratch)  vs  all-MiniLM-L6-v2 (pretrained)  — same eval harness")
    print("=" * 78)
    print(f"  {'model':<22} {'R@1':>7} {'MRR':>6} {'realSim':>7} {'noneSim':>7}   "
          f"{'honest':>7} {'noneRec':>8}")
    print(_row(f"ours ({ours_params/1e6:.2f}M)", ours))
    print(_row(f"MiniLM ({mini_params/1e6:.0f}M)", mini))

    print("\n  per-intent R@1:")
    print(f"    {'intent':<14} {'ours':>7} {'MiniLM':>8}")
    for it in ours["intents"]:
        print(f"    {it:<14} {ours['per_intent'][it]['recall@1']*100:6.1f}% "
              f"{mini['per_intent'][it]['recall@1']*100:7.1f}%")

    print(f"\n  read: 'honest' = overall acc (real+none) at the val-tuned τ; "
          f"'noneSim' lower = junk pushed further from prototypes.")


if __name__ == "__main__":
    main()
