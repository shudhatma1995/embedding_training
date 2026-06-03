"""
train.py  -  Stage 4: contrastive training for the assistant intent classifier.
================================================================================

What this does
  Learns sentence embeddings such that utterances of the SAME intent land close
  together and DIFFERENT intents land apart. We reuse the embedder + tokenizer
  from `customer_intent_search` UNTOUCHED (so upstream stays cleanly mergeable)
  and only write the small, intent-specific training pieces here.

The contrastive idea (Multiple-Negatives-Ranking / InfoNCE)
  A "positive pair" = two DIFFERENT phrasings of the same intent
  (e.g. "play some jazz" / "put on my workout playlist").  In a batch we embed
  all anchors A and all positives P, form the similarity matrix S = A @ Pᵀ, and
  ask: for each anchor (row), is its OWN positive (the diagonal) the most similar?
  Cross-entropy with the diagonal as the target pushes the diagonal up and every
  off-diagonal (a different intent = a free negative) down.

Why `none` is NOT trained here
  `none` is "everything else" (gibberish, statements, out-of-scope). It is not a
  tight semantic cluster, so pairing none-with-none as positives would teach
  nonsense. Per intents.py we train the 4 REAL intents as prototypes and decide
  `none` with a similarity THRESHOLD at evaluation time (Stage 5).

Why batch size == number of intents
  We put exactly ONE pair per intent in each batch (IntentAwareBatchSampler idea).
  That guarantees the off-diagonal entries are always DIFFERENT intents — i.e. no
  "false negatives" where we'd wrongly punish a correct match. The cost: with only
  4 intents, each anchor sees just 3 negatives (a weak signal). That's an honest
  limitation of few-class contrastive training and a good lever to revisit later
  (more intents, supervised-contrastive masking, or hard-negative mining).

Run:  python train.py            (writes models/finetuned/{model.pt,tokenizer.json,config.json})
"""
import os
import sys
import json
import time
import random
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

# Reuse the embedder + tokenizer from customer_intent_search WITHOUT editing them.
# Both modules depend only on torch (no sentence_transformers / sklearn / matplotlib),
# so they import cleanly in this minimal venv.
_CIS = os.path.join(os.path.dirname(__file__), "..", "customer_intent_search")
sys.path.insert(0, _CIS)
from tokenizer import SimpleTokenizer   # noqa: E402
from model import build_model           # noqa: E402

from shared import load_grouped, build_prototypes   # noqa: E402  (shared helpers)

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")

# Intents trained as prototypes. `none` is intentionally excluded (threshold at eval).
TRAIN_INTENTS = ["answers", "media", "smart_home", "productivity"]


# ── data ─────────────────────────────────────────────────────────────────────
def make_pairs(groups: dict, pairs_per_intent: int, rng: random.Random):
    """
    One positive pair = two DIFFERENT utterances of the same intent.
    Re-sampled every epoch (dynamic pairing) so a query meets many partners.
    Returns parallel lists: anchors, positives, intents.
    """
    A, P, I = [], [], []
    for intent, qs in groups.items():
        s = qs[:]
        rng.shuffle(s)
        n = min(pairs_per_intent, len(s) // 2)
        for i in range(0, n * 2, 2):
            A.append(s[i])
            P.append(s[i + 1])
            I.append(intent)
    return A, P, I


def iter_batches(A, P, I, rng: random.Random):
    """
    Yield batches with EXACTLY ONE pair per intent (no false negatives).
    Each yield: (anchor_texts, positive_texts) of length == n_intents.
    Incomplete trailing batches are dropped to keep the loss matrix square.
    """
    by = defaultdict(list)
    for idx, intent in enumerate(I):
        by[intent].append(idx)                 # group pair-indices by intent
    for v in by.values():
        rng.shuffle(v)
    n_intents = len(by)
    rounds = max(len(v) for v in by.values())
    for r in range(rounds):
        batch = [v[r] for v in by.values() if r < len(v)]
        if len(batch) == n_intents:           # full batch only
            yield [A[i] for i in batch], [P[i] for i in batch]


# ── loss ─────────────────────────────────────────────────────────────────────
def mnr_loss(anchor_emb: torch.Tensor, pos_emb: torch.Tensor, temperature: float,
             neg_emb: torch.Tensor = None):
    """
    Multiple-Negatives-Ranking / InfoNCE.
        S = (A @ Pᵀ) / τ          # [B,B]; row i = anchor i vs every positive
        target_i = i              # the diagonal is the correct match
        loss = cross_entropy(S, targets)
    Low τ sharpens the distribution (harder signal); high τ softens it.

    none-as-negatives (neg_emb [k,D]): append k SHARED junk negatives as extra
    columns → S grows [B,B] → [B,B+k]. Targets are unchanged (still the diagonal),
    so every real anchor must rank its positive above all other intents AND above
    every `none` example. This pushes junk AWAY from all prototypes.
    """
    sim = (anchor_emb @ pos_emb.T) / temperature          # [B,B]
    if neg_emb is not None and len(neg_emb):
        sim = torch.cat([sim, (anchor_emb @ neg_emb.T) / temperature], dim=1)  # [B,B+k]
    labels = torch.arange(anchor_emb.shape[0], device=sim.device)
    return F.cross_entropy(sim, labels)


# ── evaluation (prototype / centroid classifier) ─────────────────────────────
@torch.no_grad()
def proto_recall1(model, tok, train_groups, test_groups, device) -> float:
    """
    Centroid classifier: prototype[intent] = L2-normalized MEAN of that intent's
    train embeddings. Each test query is labelled by its nearest prototype (cosine).
    Returns Recall@1 over the held-out test queries (the 4 real intents).

    This is the honest generalization signal: test.json uses unseen phrasings AND
    unseen entities, so this number reflects pattern-learning, not memorization.
    """
    model.eval()
    protos, intents = build_prototypes(model, tok, train_groups, device)  # [C,D], [C]

    correct = total = 0
    for ti, it in enumerate(intents):
        q = model.encode(test_groups[it], tok, device=device)    # [m,D]
        if len(q) == 0:
            continue
        pred = (q @ protos.T).argmax(axis=1)                     # nearest prototype
        correct += int((pred == ti).sum())
        total += len(q)
    model.train()
    return correct / max(1, total)


# ── scheduler ────────────────────────────────────────────────────────────────
def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup 0→1 over warmup_steps, then linear decay to 0.1×."""
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 1.0 - 0.9 * progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── training ─────────────────────────────────────────────────────────────────
def embed(model, tok, texts, device):
    """Tokenize `texts`, move to device, run the model forward.
    Returns [len(texts), D] L2-normalized embeddings (grad-enabled — this is the
    training forward, unlike model.encode which is the no-grad numpy path)."""
    ids, mask = tok.encode_batch(texts)
    return model(ids.to(device), mask.to(device))


def run_epoch(model, tok, train_groups, none_pool, optimizer, scheduler,
              pairs_per_intent, temperature, none_neg_k, rng, device):
    """One training epoch: fresh pairs → one-pair-per-intent batches → step.
    Returns (mean_loss, train_acc) for the epoch log."""
    A, P, I = make_pairs(train_groups, pairs_per_intent, rng)   # dynamic each epoch
    model.train()
    tot_loss = correct = npairs = nbatch = 0
    for anchors, positives in iter_batches(A, P, I, rng):
        ae = embed(model, tok, anchors, device)
        pe = embed(model, tok, positives, device)

        # sample k shared `none` negatives for this batch (if enabled)
        neg_emb = None
        if none_neg_k > 0:
            neg_texts = [rng.choice(none_pool) for _ in range(none_neg_k)]
            neg_emb = embed(model, tok, neg_texts, device)

        loss = mnr_loss(ae, pe, temperature, neg_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        tot_loss += loss.item()
        nbatch += 1
        with torch.no_grad():
            pred = (ae @ pe.T).argmax(dim=1)
            labels = torch.arange(ae.shape[0], device=device)
            correct += int((pred == labels).sum())
            npairs += ae.shape[0]
    return tot_loss / max(1, nbatch), correct / max(1, npairs)


def save_artifacts(output_dir, model, tok, config):
    """Write model.pt, tokenizer.json, config.json to output_dir; return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "model": os.path.join(output_dir, "model.pt"),
        "tokenizer": os.path.join(output_dir, "tokenizer.json"),
        "config": os.path.join(output_dir, "config.json"),
    }
    torch.save(model.state_dict(), paths["model"])
    tok.save(paths["tokenizer"])
    with open(paths["config"], "w") as f:
        json.dump(config, f, indent=2)
    return paths


def train(args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # data: 4 real intents only (none excluded — handled by threshold in Stage 5)
    train_groups = load_grouped(os.path.join(DATA_DIR, "train.json"), TRAIN_INTENTS)
    test_groups = load_grouped(os.path.join(DATA_DIR, "test.json"), TRAIN_INTENTS)
    n_intents = len(train_groups)
    print("\n  Train queries per intent:")
    for it in TRAIN_INTENTS:
        print(f"    {it:<13} {len(train_groups[it]):>4} train | {len(test_groups[it]):>3} test")

    # `none` enters training via TWO independent switches (so we can ablate them):
    #   --none-in-vocab : put junk words in the tokenizer vocab (vs [UNK])
    #   --none-neg-k    : use junk as shared negatives in the loss
    # Negatives are meaningless if their words aren't in vocab, so k>0 forces in-vocab.
    none_in_vocab = args.none_in_vocab or args.none_neg_k > 0
    none_pool = []
    if none_in_vocab:
        none_pool = load_grouped(os.path.join(DATA_DIR, "train.json"), ["none"])["none"]

    # tokenizer fit on training texts (+ none pool iff none_in_vocab)
    fit_texts = [t for qs in train_groups.values() for t in qs]
    if none_in_vocab:
        fit_texts = fit_texts + none_pool
    tok = SimpleTokenizer(max_vocab_size=8000, max_seq_len=32).fit(fit_texts)
    print(f"\n  none in vocab : {none_in_vocab}  ({len(none_pool)} junk examples in fit)")
    if args.none_neg_k > 0:
        print(f"  none negatives: k={args.none_neg_k} per batch")
    print(f"  Vocab size    : {tok.vocab_size}")

    model = build_model(tok).to(device)
    print(f"  Parameters : {model.n_parameters():,}")

    # one pair per intent per batch → batch_size == n_intents
    batches_per_epoch = args.pairs_per_intent          # (n_intents pairs each → n_intents-sized batches)
    total_steps = batches_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = make_scheduler(optimizer, args.warmup_steps, total_steps)

    # BEFORE-training baseline (random init): shows what training actually buys
    r1_before = proto_recall1(model, tok, train_groups, test_groups, device)
    print(f"\n  test Recall@1 (random init, before training): {r1_before*100:.1f}%")
    print(f"\n  Training for {args.epochs} epochs "
          f"(batch={n_intents}, τ={args.temperature}, lr={args.lr})...")
    print(f"  {'epoch':>5} | {'loss':>7} | {'train_acc':>9} | {'test_R@1':>8} | {'time':>5}")
    print("  " + "─" * 50)

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, tok, train_groups, none_pool, optimizer, scheduler,
            args.pairs_per_intent, args.temperature, args.none_neg_k, rng, device)
        test_r1 = proto_recall1(model, tok, train_groups, test_groups, device)
        history.append({"loss": train_loss, "train_acc": train_acc, "test_r1": test_r1})
        print(f"  {epoch:>5} | {train_loss:>7.4f} | {train_acc*100:>8.1f}% | "
              f"{test_r1*100:>7.1f}% | {time.time()-t0:>4.1f}s")

    best = max(history, key=lambda h: h["test_r1"])
    best_ep = history.index(best) + 1
    print(f"\n  best test Recall@1: {best['test_r1']*100:.1f}% at epoch {best_ep} "
          f"(random-init baseline was {r1_before*100:.1f}%)")

    # ── save artifacts ─────────────────────────────────────────
    paths = save_artifacts(args.output_dir, model, tok, {
        "train_intents": TRAIN_INTENTS,
        "epochs": args.epochs,
        "batch_size": n_intents,
        "pairs_per_intent": args.pairs_per_intent,
        "lr": args.lr,
        "temperature": args.temperature,
        "none_neg_k": args.none_neg_k,
        "none_in_vocab": none_in_vocab,
        "vocab_size": tok.vocab_size,
        "r1_random_init": r1_before,
        "history": history,
    })
    print(f"\n  Model     → {paths['model']}")
    print(f"  Tokenizer → {paths['tokenizer']}")
    print(f"  Config    → {paths['config']}")
    return model, tok


def main():
    ap = argparse.ArgumentParser(description="Train the assistant intent embedder")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--pairs-per-intent", type=int, default=30,
                    help="positive pairs sampled per intent per epoch (max = n_queries // 2)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--none-neg-k", type=int, default=0,
                    help="shared `none` negatives appended per batch (0 = off)")
    ap.add_argument("--none-in-vocab", action="store_true",
                    help="fit tokenizer on the none pool too (junk words in-vocab vs [UNK])")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(HERE, "models", "finetuned"))
    args = ap.parse_args()
    print("=" * 60)
    print("TRAINING assistant intent embedder (Stage 4)")
    print("=" * 60)
    train(args)


if __name__ == "__main__":
    main()
