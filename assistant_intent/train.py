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
  nonsense. Per intents.py we train the REAL intents as prototypes and decide
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

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F

# Embedder + tokenizer come from customer_intent_search (reused untouched) via cis.py.
from cis import SimpleTokenizer, build_model

# Side code (I/O + batch shaping) in data.py; classifier kernel in shared.py.
from data import DATA_DIR, intents_in_file, iter_batches, load_by_intent, make_pairs, save_artifacts

# Intent taxonomy is owned by intents.py (single source of truth); `none` is excluded
# from the trainable set by construction.
from intents import NONE_ID, REAL_INTENTS
from shared import make_encoder, mine_hard_negatives, proto_recall1

HERE = os.path.dirname(__file__)


def resolve_train_intents(requested, data_path):
    """Pick & validate the intents to train as prototypes.

    requested=None → all of REAL_INTENTS (the default). Otherwise the user's subset.
    Fails loudly (rather than silently training on nothing) if the request:
      - includes `none`     → it's the out-of-scope class, never a positive prototype;
      - names an intent that isn't actually in the data file;
      - has fewer than 2 intents → contrastive loss would have no in-batch negatives.
    """
    chosen = list(requested) if requested else list(REAL_INTENTS)
    present = set(intents_in_file(data_path))
    if NONE_ID in chosen:
        raise SystemExit(
            f"`{NONE_ID}` can't be trained as a prototype (it's the out-of-scope class, "
            f"handled by threshold/negatives). Remove it from --intents."
        )
    missing = [i for i in chosen if i not in present]
    if missing:
        raise SystemExit(
            f"--intents {missing} not found in {os.path.basename(data_path)}. "
            f"Available: {sorted(present - {NONE_ID})}"
        )
    if len(chosen) < 2:
        raise SystemExit(
            f"Need ≥2 intents for contrastive training (got {chosen}); with one intent "
            f"the similarity matrix is 1×1 and the loss has no in-batch negatives."
        )
    return chosen


# ── loss ─────────────────────────────────────────────────────────────────────
def mnr_loss(
    anchor_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    temperature: float,
    neg_emb: torch.Tensor = None,
    hard_neg_emb: torch.Tensor = None,
):
    """
    Multiple-Negatives-Ranking / InfoNCE.
        S = (A @ Pᵀ) / τ          # [B,B]; row i = anchor i vs every positive
        target_i = i              # the diagonal is the correct match
        loss = cross_entropy(S, targets)
    Low τ sharpens the distribution (harder signal); high τ softens it.

    Two kinds of extra negative columns can be appended; targets stay the diagonal,
    so the final matrix is [B, B + k_none + k_hn] and every anchor must rank its own
    positive above ALL of them:

    none-as-negatives (neg_emb [k,D]): k SHARED junk vectors — the SAME columns for
    every row, so a plain matmul `A @ neg_embᵀ → [B,k]` suffices. Pushes junk away
    from all prototypes.

    hard negatives (hard_neg_emb [B*k, D]): PER-ANCHOR — row i gets its OWN k mined
    confusers (the most-similar different-intent queries, from mine_hard_negatives).
    Because each row's columns differ we can't share a matmul; reshape to [B,k,D] and
    use a batched matmul so anchor i is dotted only with ITS OWN k negatives:
        [B,1,D] @ [B,D,k] → [B,1,k] → [B,k]
    This sharpens exactly the confusable boundaries (e.g. media vs smart_home) that
    in-batch negatives leave alone.
    """
    sim = (anchor_emb @ pos_emb.T) / temperature  # [B,B]
    if neg_emb is not None and len(neg_emb):
        sim = torch.cat([sim, (anchor_emb @ neg_emb.T) / temperature], dim=1)  # [B,B+k_none]
    if hard_neg_emb is not None and len(hard_neg_emb):
        B, D = anchor_emb.shape
        k = hard_neg_emb.shape[0] // B  # k per anchor, inferred from [B*k, D]
        hn = hard_neg_emb.view(B, k, D)  # [B,k,D]: each anchor's own k negatives
        hard_sim = torch.bmm(anchor_emb.unsqueeze(1), hn.transpose(1, 2)).squeeze(1)  # [B,k]
        sim = torch.cat([sim, hard_sim / temperature], dim=1)  # [B, B+k_none+k_hn]
    labels = torch.arange(anchor_emb.shape[0], device=sim.device)
    return F.cross_entropy(sim, labels)


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


def run_epoch(
    model,
    tok,
    train_groups,
    none_pool,
    optimizer,
    scheduler,
    pairs_per_intent,
    temperature,
    none_neg_k,
    rng,
    device,
    hard_neg_map=None,
    hn_top_k=0,
):
    """One training epoch: fresh pairs → one-pair-per-intent batches → step.
    Returns (mean_loss, train_acc) for the epoch log.

    hard_neg_map ({intent: [texts]} from mine_hard_negatives) + hn_top_k>0 enable
    per-anchor hard negatives: for each batch we sample k confusers from each
    anchor's pool (in row order) and hand them to mnr_loss as extra columns."""
    A, P, I = make_pairs(train_groups, pairs_per_intent, rng)  # dynamic each epoch
    model.train()
    tot_loss = correct = npairs = nbatch = 0
    for anchors, positives, intents in iter_batches(A, P, I, rng):
        ae = embed(model, tok, anchors, device)
        pe = embed(model, tok, positives, device)

        # sample k shared `none` negatives for this batch (if enabled)
        neg_emb = None
        if none_neg_k > 0:
            neg_texts = [rng.choice(none_pool) for _ in range(none_neg_k)]
            neg_emb = embed(model, tok, neg_texts, device)

        # sample each anchor's OWN k hard negatives (row order matches `intents`)
        hard_emb = None
        if hard_neg_map is not None and hn_top_k > 0:
            hard_texts = [rng.choice(hard_neg_map[it]) for it in intents for _ in range(hn_top_k)]
            hard_emb = embed(model, tok, hard_texts, device)

        loss = mnr_loss(ae, pe, temperature, neg_emb, hard_emb)

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


def train(args):
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # which real intents to train as prototypes (validated subset; none excluded —
    # handled by threshold in Stage 5). Default = all of REAL_INTENTS.
    train_intents = resolve_train_intents(args.intents, os.path.join(DATA_DIR, "train.json"))
    train_groups = load_by_intent(os.path.join(DATA_DIR, "train.json"), train_intents)
    test_groups = load_by_intent(os.path.join(DATA_DIR, "test.json"), train_intents)
    n_intents = len(train_groups)
    print(f"\n  Training intents ({n_intents}): {', '.join(train_intents)}")
    print("  Train queries per intent:")
    for it in train_intents:
        print(f"    {it:<13} {len(train_groups[it]):>4} train | {len(test_groups[it]):>3} test")

    # `none` enters training via TWO independent switches (so we can ablate them):
    #   --none-in-vocab : put junk words in the tokenizer vocab (vs [UNK])
    #   --none-neg-k    : use junk as shared negatives in the loss
    # Negatives are meaningless if their words aren't in vocab, so k>0 forces in-vocab.
    none_in_vocab = args.none_in_vocab or args.none_neg_k > 0
    none_pool = []
    if none_in_vocab:
        none_pool = load_by_intent(os.path.join(DATA_DIR, "train.json"), [NONE_ID])[NONE_ID]

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
    encode = make_encoder(model, tok, device)  # Encoder for the in-loop proto metric

    # one pair per intent per batch → batch_size == n_intents
    batches_per_epoch = args.pairs_per_intent  # (n_intents pairs each → n_intents-sized batches)
    total_steps = batches_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = make_scheduler(optimizer, args.warmup_steps, total_steps)

    # BEFORE-training baseline (random init): shows what training actually buys
    r1_before = proto_recall1(encode, train_groups, test_groups)
    print(f"\n  test Recall@1 (random init, before training): {r1_before * 100:.1f}%")
    print(
        f"\n  Training for {args.epochs} epochs "
        f"(batch={n_intents}, τ={args.temperature}, lr={args.lr})..."
    )
    # Hard-negative knobs are optional (default off) so a hand-built args Namespace
    # — e.g. the smoke test — works without them; build_parser supplies the defaults.
    hn_top_k = getattr(args, "hn_top_k", 0)
    hn_start = getattr(args, "hn_start_epoch", 3)
    hn_refresh = getattr(args, "hn_refresh_every", 3)
    if hn_top_k > 0:
        print(
            f"  Hard negatives: top_k={hn_top_k}, start_epoch={hn_start}, "
            f"refresh_every={hn_refresh} (warmup uses in-batch negs only)"
        )
    print(f"  {'epoch':>5} | {'loss':>7} | {'train_acc':>9} | {'test_R@1':>8} | {'time':>5}")
    print("  " + "─" * 50)

    history = []
    hard_neg_map = None  # None until mining kicks in (after warmup)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # (Re-)mine hard negatives with CURRENT weights at the start of the warmup
        # epoch and every refresh interval after. `encode` reads live model weights.
        if hn_top_k > 0 and epoch >= hn_start and (epoch - hn_start) % hn_refresh == 0:
            hard_neg_map = mine_hard_negatives(encode, train_groups, top_k=hn_top_k)
        train_loss, train_acc = run_epoch(
            model,
            tok,
            train_groups,
            none_pool,
            optimizer,
            scheduler,
            args.pairs_per_intent,
            args.temperature,
            args.none_neg_k,
            rng,
            device,
            hard_neg_map,
            hn_top_k,
        )
        test_r1 = proto_recall1(encode, train_groups, test_groups)
        history.append({"loss": train_loss, "train_acc": train_acc, "test_r1": test_r1})
        print(
            f"  {epoch:>5} | {train_loss:>7.4f} | {train_acc * 100:>8.1f}% | "
            f"{test_r1 * 100:>7.1f}% | {time.time() - t0:>4.1f}s"
        )

    best = max(history, key=lambda h: h["test_r1"])
    best_ep = history.index(best) + 1
    print(
        f"\n  best test Recall@1: {best['test_r1'] * 100:.1f}% at epoch {best_ep} "
        f"(random-init baseline was {r1_before * 100:.1f}%)"
    )

    # ── save artifacts ─────────────────────────────────────────
    paths = save_artifacts(
        args.output_dir,
        model,
        tok,
        {
            "train_intents": train_intents,
            "epochs": args.epochs,
            "batch_size": n_intents,
            "pairs_per_intent": args.pairs_per_intent,
            "lr": args.lr,
            "temperature": args.temperature,
            "none_neg_k": args.none_neg_k,
            "none_in_vocab": none_in_vocab,
            "hn_top_k": hn_top_k,
            "hn_start_epoch": hn_start if hn_top_k > 0 else None,
            "hn_refresh_every": hn_refresh if hn_top_k > 0 else None,
            "vocab_size": tok.vocab_size,
            "r1_random_init": r1_before,
            "history": history,
        },
    )
    print(f"\n  Model     → {paths['model']}")
    print(f"  Tokenizer → {paths['tokenizer']}")
    print(f"  Config    → {paths['config']}")
    return model, tok


def build_parser():
    """The training CLI. Exposed so other entry points (e.g. experiments.py) can
    do `build_parser().parse_args([])` to get the exact default hyperparameters
    instead of re-typing them — one source of truth for the defaults."""
    ap = argparse.ArgumentParser(description="Train the assistant intent embedder")
    ap.add_argument(
        "--intents",
        nargs="+",
        default=None,
        help="subset of real intents to train (default: all of REAL_INTENTS). "
        "`none` is not allowed; needs ≥2 intents.",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument(
        "--pairs-per-intent",
        type=int,
        default=30,
        help="positive pairs sampled per intent per epoch (max = n_queries // 2)",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument(
        "--none-neg-k",
        type=int,
        default=0,
        help="shared `none` negatives appended per batch (0 = off)",
    )
    ap.add_argument(
        "--none-in-vocab",
        action="store_true",
        help="fit tokenizer on the none pool too (junk words in-vocab vs [UNK])",
    )
    ap.add_argument(
        "--hn-top-k",
        type=int,
        default=0,
        help="per-anchor hard negatives mined per batch (0 = off). Each anchor gets "
        "its own k most-confusable different-intent queries as extra loss columns.",
    )
    ap.add_argument(
        "--hn-start-epoch",
        type=int,
        default=3,
        help="epoch to begin mining (earlier epochs warm up on in-batch negatives only, "
        "so the first mining runs on a non-random model).",
    )
    ap.add_argument(
        "--hn-refresh-every",
        type=int,
        default=3,
        help="re-mine hard negatives every N epochs with current weights "
        "(yesterday's hard negs become easy as the model sharpens).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default=os.path.join(HERE, "models", "finetuned"))
    return ap


def main():
    args = build_parser().parse_args()
    print("=" * 60)
    print("TRAINING assistant intent embedder (Stage 4)")
    print("=" * 60)
    train(args)


if __name__ == "__main__":
    main()
