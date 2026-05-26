"""
Fine-tuning Script: MiniIntent Embedding Model
================================================
Trains a small transformer embedding model from scratch on the customer
intent dataset using MultipleNegativesRankingLoss (in-batch negatives).

Two training modes:
  1. From scratch: randomly initialised transformer → fine-tuned on intent pairs
  2. Baseline: same architecture, ZERO gradient updates (for fair comparison)

Training strategy:
  - Positive pairs: two queries of the same intent class
  - Negatives: all other queries in the batch (in-batch negatives)
  - Loss: MultipleNegativesRankingLoss ≡ cross-entropy on similarity logits
  - With batch size B, each positive has B-1 hard in-batch negatives

Model: MiniIntentEmbedder
  - Architecture: 2-layer transformer encoder, 4 attention heads
  - Embedding dim: 128
  - Parameters: ~600k (tiny!)
  - Input: word-tokenised text (max 32 tokens)

Usage:
    python train.py [--epochs 15] [--batch-size 64]
"""
import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_preparation import load_clinc_dataset, create_training_pairs
from model import MiniIntentEmbedder, MiniIntentConfig, build_model
from tokenizer import SimpleTokenizer
from utils import set_seed, print_banner


# ── Loss function ─────────────────────────────────────────────────────────────

class MultipleNegativesRankingLoss(nn.Module):
    """
    MultipleNegativesRankingLoss (MNR Loss / InfoNCE).

    Given a batch of (anchor, positive) pairs, treats all other positives
    in the batch as negatives for each anchor.

    Loss = -mean(log softmax(sim(anchor_i, positive_i) / temp) for all i)

    Reference: Henderson et al., 2017 "Efficient Natural Language Response
    Suggestion for Smart Reply"
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_embs: torch.Tensor,    # (B, D)
        positive_embs: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        # Both are already L2-normalised by the model
        # sim[i, j] = cosine_similarity(anchor_i, positive_j)
        sim = torch.matmul(anchor_embs, positive_embs.T) / self.temperature  # (B, B)

        # Labels: diagonal is the correct positive
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss = F.cross_entropy(sim, labels)
        return loss


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """Dataset of (anchor_text, positive_text) intent pairs."""

    def __init__(self, pairs: list):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return pair.texts[0], pair.texts[1]


def collate_fn(tokenizer: SimpleTokenizer):
    """Returns a collate function that tokenises text pairs."""
    def _collate(batch):
        anchors = [item[0] for item in batch]
        positives = [item[1] for item in batch]
        anchor_ids, anchor_mask = tokenizer.encode_batch(anchors)
        positive_ids, positive_mask = tokenizer.encode_batch(positives)
        return anchor_ids, anchor_mask, positive_ids, positive_mask
    return _collate


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: MiniIntentEmbedder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: MultipleNegativesRankingLoss,
    device: str,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (mean_loss, accuracy)."""
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for anchor_ids, anchor_mask, pos_ids, pos_mask in loader:
        anchor_ids = anchor_ids.to(device)
        anchor_mask = anchor_mask.to(device)
        pos_ids = pos_ids.to(device)
        pos_mask = pos_mask.to(device)

        anchor_embs = model(anchor_ids, anchor_mask)
        pos_embs = model(pos_ids, pos_mask)

        loss = loss_fn(anchor_embs, pos_embs)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Accuracy: fraction of samples where the correct positive ranks first
        with torch.no_grad():
            sim = torch.matmul(anchor_embs, pos_embs.T)
            correct = (sim.argmax(dim=1) == torch.arange(len(anchor_embs), device=device))
            acc = correct.float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MiniIntentEmbedder on customer intent data"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pairs-per-intent", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="../models/finetuned")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args=None):
    if args is None:
        args = parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print_banner("Training MiniIntent Embedding Model")
    print(f"  Architecture:    2-layer transformer, 128-dim embeddings")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Temperature:     {args.temperature}")
    print(f"  Device:          {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print_banner("Step 1: Data Preparation")
    train_groups, val_groups, _ = load_clinc_dataset()
    train_pairs = create_training_pairs(
        train_groups,
        pairs_per_intent=args.pairs_per_intent,
        seed=args.seed
    )

    # Collect all texts for tokenizer vocabulary
    all_texts = []
    for queries in train_groups.values():
        all_texts.extend(queries)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print_banner("Step 2: Building Tokenizer")
    tokenizer = SimpleTokenizer(max_vocab_size=6000, max_length=32)
    tokenizer.fit(all_texts)
    print(f"  {tokenizer}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print_banner("Step 3: Building Model")
    model = build_model(tokenizer).to(device)
    n_params = model.n_parameters()
    print(f"  MiniIntentEmbedder")
    print(f"  Parameters:   {n_params:,} ({n_params/1e6:.3f}M)")
    print(f"  Embed dim:    {model.config.embed_dim}")
    print(f"  Layers:       {model.config.n_layers}")
    print(f"  Heads:        {model.config.n_heads}")
    print(f"  Vocab size:   {tokenizer.vocab_size}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset = PairDataset(train_pairs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer),
        drop_last=True,
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps = len(loader) * args.epochs
    warmup = args.warmup_steps

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 1.0 - progress)  # linear decay to 10% LR

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = MultipleNegativesRankingLoss(temperature=args.temperature)

    # ── Training loop ─────────────────────────────────────────────────────────
    print_banner("Step 4: Training")
    print(f"  {'Epoch':<8} {'Loss':>10} {'Acc':>10}")
    print("  " + "-" * 30)

    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, scheduler,
                                    loss_fn, device)
        history.append({"epoch": epoch, "loss": loss, "acc": acc})
        print(f"  {epoch:<8} {loss:>10.4f} {acc:>9.2%}")

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    print_banner("Step 5: Saving Model")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    config_dict = {
        "model_config": {
            "vocab_size": model.config.vocab_size,
            "embed_dim": model.config.embed_dim,
            "n_heads": model.config.n_heads,
            "n_layers": model.config.n_layers,
            "ffn_dim": model.config.ffn_dim,
            "max_seq_len": model.config.max_seq_len,
            "dropout": model.config.dropout,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "pairs_per_intent": args.pairs_per_intent,
            "n_training_pairs": len(train_pairs),
            "training_time_seconds": round(elapsed, 1),
        },
        "training_history": history,
    }

    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"  Model saved to: {args.output_dir}/")
    return model, tokenizer, args.output_dir


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer, model_dir = train(args)
    print(f"\nDone! Run: python evaluate.py --model-dir {model_dir}")
