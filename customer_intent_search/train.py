import argparse
import os
import time
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tokenizer import SimpleTokenizer
from model import build_model
from data_preparation import load_dataset, create_training_pairs, IntentAwareBatchSampler

class MultipleNegativesRankingLoss(nn.Module):
    """
    Contrastive loss using all other positives in the batch as negatives.

    For a batch of (anchor, positive) pairs:
      - Embed all anchors   → matrix A shape (batch_size, 128)
      - Embed all positives → matrix P shape (batch_size, 128)
      - Similarity matrix   S = A @ P.T    shape (batch_size, batch_size)
      - Diagonal            = correct matches  → should be HIGH
      - Off-diagonal        = wrong matches    → should be LOW
      - Loss = cross entropy treating diagonal as the correct class
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        # temperature scales scores before softmax
        # low  τ = 0.05 → scores more peaked → harder training signal
        # high τ = 1.00 → scores flatter     → softer training signal
        self.temperature = temperature

    def forward(
        self,
        anchor_emb: torch.Tensor,    # (batch_size, 128)
        positive_emb: torch.Tensor,  # (batch_size, 128)
    ) -> torch.Tensor:

        # similarity matrix: every anchor vs every positive
        # sim[i][j] = similarity of anchor_i with positive_j
        # shape (batch_size, batch_size)
        sim = torch.matmul(anchor_emb, positive_emb.T) / self.temperature

        # correct label for each anchor = its own index
        # anchor_0 → positive_0, anchor_1 → positive_1, etc.
        # e.g. labels = [0, 1, 2, ..., batch_size-1]
        labels = torch.arange(sim.shape[0], device=sim.device)

        # cross entropy:
        # treats each row of sim as logits over all positives
        # correct class = diagonal index
        # loss is low  when sim[i][i] >> sim[i][j] for all j≠i
        # loss is high when sim[i][i] is not the highest in row i
        loss = F.cross_entropy(sim, labels)

        return loss


class PairDataset(Dataset):
    """
    Wraps a list of InputExample pairs for PyTorch DataLoader.
    Each item returns (anchor_text, positive_text).
    """

    def __init__(self, pairs):
        # pairs = list of InputExample objects
        # e.g. [InputExample(["where is my order", "track my package"]), ...]
        self.pairs = pairs

    def __len__(self):
        # tells DataLoader how many items exist in total
        # e.g. 400 pairs → 400
        return len(self.pairs)

    def __getitem__(self, idx):
        # DataLoader calls this with a specific index
        # returns the two texts for that pair as a tuple
        # e.g. idx=0 → ("where is my order?", "track my package")
        anchor   = self.pairs[idx].texts[0]
        positive = self.pairs[idx].texts[1]
        return anchor, positive


def make_collate_fn(tokenizer):
    """
    Returns a function that tokenizes a batch of (anchor, positive) pairs.

    Why a closure?
    DataLoader expects collate_fn to take only one argument (the batch).
    But we also need the tokenizer inside. A closure captures the tokenizer
    in the outer scope so the inner function can use it without being passed
    as an argument.

    DataLoader calls collate_fn with:
        batch = [
            ("where is my order",  "track my package"),      ← item 0
            ("cancel my order",    "I want to cancel"),      ← item 1
            ("where is my refund", "refund not received"),   ← item 2
            ...
        ]

    Returns four tensors:
        anchor_ids:    (batch_size, max_seq_len)
        anchor_mask:   (batch_size, max_seq_len)
        positive_ids:  (batch_size, max_seq_len)
        positive_mask: (batch_size, max_seq_len)
    """

    def collate_fn(batch):
        # batch is a list of (anchor_text, positive_text) tuples
        # separate anchors and positives into two lists
        # e.g. anchors   = ["where is my order",  "cancel my order", ...]
        # e.g. positives = ["track my package",   "I want to cancel", ...]
        anchors   = [item[0] for item in batch]
        positives = [item[1] for item in batch]

        # tokenize anchors → two tensors each shape (batch_size, max_seq_len)
        # anchor_ids:  token IDs  e.g. [[2, 45, 12, 0, ...], [2, 33, 0, ...]]
        # anchor_mask: 1=real token, 0=padding
        anchor_ids,   anchor_mask   = tokenizer.encode_batch(anchors)

        # tokenize positives → same structure
        positive_ids, positive_mask = tokenizer.encode_batch(positives)

        return anchor_ids, anchor_mask, positive_ids, positive_mask

    # return the inner function — this is what DataLoader will call
    return collate_fn


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Learning rate schedule: linear warmup then linear decay.

    warmup phase  (steps 0 → warmup_steps):
        lr increases linearly from 0 → initial_lr

    decay phase   (steps warmup_steps → total_steps):
        lr decreases linearly from initial_lr → 0.1 * initial_lr

    Why warmup?
        At the start weights are random. A large lr causes huge unstable
        updates. Warming up gradually lets the model settle before applying
        the full learning rate.

    Why decay?
        As training progresses the model is close to a good solution.
        Smaller steps prevent overshooting and help fine convergence.
    """

    def lr_lambda(current_step: int):

        # warmup phase: ramp from 0.0 → 1.0
        # e.g. warmup_steps=50, current_step=25 → 25/50 = 0.5 → 50% of lr
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        # decay phase: ramp from 1.0 → 0.1
        # progress goes from 0.0 (start of decay) to 1.0 (end of training)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)

        # at progress=0.0 → lr = 1.0 * initial_lr
        # at progress=1.0 → lr = 0.1 * initial_lr
        # never drops below 10% of initial lr
        return max(0.1, 1.0 - 0.9 * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epoch: int,
) -> dict:
    """
    Run one full pass over all training batches.

    One epoch = model sees every training pair exactly once.
    Returns avg loss and accuracy for this epoch.

    for each batch:

    1. FORWARD
        anchor_ids → model → anchor_emb   (batch_size, 128)
        positive_ids → model → positive_emb (batch_size, 128)
        loss = cross_entropy(sim_matrix)

    2. BACKWARD
        optimizer.zero_grad()   ← clear old gradients
        loss.backward()         ← compute new gradients
        clip_grad_norm_()       ← prevent explosions

    3. UPDATE
        optimizer.step()        ← adjust weights
        scheduler.step()        ← adjust learning rate

    4. METRICS
        track loss and accuracy for reporting
    """

    # set model to training mode
    # enables dropout (randomly zeros activations during training)
    # dropout is disabled during evaluation via model.eval()
    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_pairs   = 0
    start_time    = time.time()

    for batch_idx, (anchor_ids, anchor_mask, positive_ids, positive_mask) in enumerate(dataloader):

        # ── move tensors to device (CPU or GPU) ───────────────
        # model and data must be on the same device
        anchor_ids    = anchor_ids.to(device)
        anchor_mask   = anchor_mask.to(device)
        positive_ids  = positive_ids.to(device)
        positive_mask = positive_mask.to(device)

        # ── forward pass ──────────────────────────────────────
        # embed anchors   → (batch_size, 128) L2 normalised vectors
        # embed positives → (batch_size, 128) L2 normalised vectors
        anchor_emb   = model(anchor_ids,   anchor_mask)
        positive_emb = model(positive_ids, positive_mask)

        # ── compute loss ──────────────────────────────────────
        # builds similarity matrix, computes cross entropy
        # loss is high when diagonal is not the highest in each row
        loss = loss_fn(anchor_emb, positive_emb)

        # ── backward pass ─────────────────────────────────────
        # clear gradients from previous batch
        # must do this before every backward() call
        # otherwise gradients accumulate across batches
        optimizer.zero_grad()

        # compute gradients for all parameters
        # traces back through the computation graph
        loss.backward()

        # clip gradients — prevents exploding gradients
        # if any gradient vector has norm > 1.0, scale it down
        # keeps updates stable, especially early in training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # update model weights using computed gradients
        optimizer.step()

        # update learning rate according to schedule
        scheduler.step()

        # ── track metrics ─────────────────────────────────────
        # accumulate loss for this batch
        total_loss += loss.item()

        # compute accuracy — no gradients needed here
        with torch.no_grad():

            # similarity matrix (batch_size, batch_size)
            sim = torch.matmul(anchor_emb, positive_emb.T)

            # for each anchor, find which positive scored highest
            # e.g. predicted = [0, 1, 2, ..., 19] if perfect
            predicted = sim.argmax(dim=1)

            # correct answer = diagonal index for each anchor
            # e.g. labels = [0, 1, 2, ..., 19]
            labels = torch.arange(sim.shape[0], device=device)

            # count how many anchors found their correct positive
            total_correct += (predicted == labels).sum().item()
            total_pairs   += sim.shape[0]

    # ── epoch summary ──────────────────────────────────────────
    elapsed  = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_pairs

    print(f"  Epoch {epoch:02d} | "
          f"loss {avg_loss:.4f} | "
          f"acc {accuracy*100:.1f}% | "
          f"time {elapsed:.1f}s")

    return {"loss": avg_loss, "accuracy": accuracy}


def plot_training_history(history: list, output_dir: str):
    """
    Generate and save training charts after training completes.

    Produces one figure with two side-by-side plots:
        Left  — Loss curve    : how loss decreased each epoch
        Right — Accuracy curve: how accuracy improved each epoch

    Args:
        history:    list of {"loss": float, "accuracy": float} per epoch
        output_dir: directory to save the chart
    """
    epochs   = list(range(1, len(history) + 1))
    losses   = [h["loss"]     for h in history]
    accs     = [h["accuracy"] * 100 for h in history]  # convert to percentage

    # create figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training History — MiniIntentEmbedder", fontsize=14, fontweight="bold")

    # ── left plot: loss curve ──────────────────────────────────
    ax1.plot(epochs, losses, color="#e74c3c", linewidth=2, marker="o", markersize=4)

    # shade area under the curve for visual appeal
    ax1.fill_between(epochs, losses, alpha=0.1, color="#e74c3c")

    # annotate first and last loss values
    ax1.annotate(f"{losses[0]:.2f}",  xy=(epochs[0],  losses[0]),
                 xytext=(8, 8),  textcoords="offset points", fontsize=9, color="#e74c3c")
    ax1.annotate(f"{losses[-1]:.4f}", xy=(epochs[-1], losses[-1]),
                 xytext=(-30, 8), textcoords="offset points", fontsize=9, color="#e74c3c")

    ax1.set_title("Loss per Epoch", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_xticks(epochs)
    ax1.grid(True, alpha=0.3)

    # ── right plot: accuracy curve ─────────────────────────────
    ax2.plot(epochs, accs, color="#2ecc71", linewidth=2, marker="o", markersize=4)

    # shade area under the curve
    ax2.fill_between(epochs, accs, alpha=0.1, color="#2ecc71")

    # annotate first and last accuracy values
    ax2.annotate(f"{accs[0]:.1f}%",  xy=(epochs[0],  accs[0]),
                 xytext=(8, -15), textcoords="offset points", fontsize=9, color="#2ecc71")
    ax2.annotate(f"{accs[-1]:.1f}%", xy=(epochs[-1], accs[-1]),
                 xytext=(-40, -15), textcoords="offset points", fontsize=9, color="#2ecc71")

    # draw a horizontal dashed line at 95% as a quality reference
    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.5, label="95% reference")
    ax2.legend(fontsize=9)

    ax2.set_title("Recall@1 Accuracy per Epoch", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # save to results directory
    plot_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot      → {plot_path}")
    return plot_path


def train(args) -> tuple:
    """
    Full training pipeline:
        1. Load dataset
        2. Build tokenizer
        3. Build model
        4. Create training pairs + custom batch sampler
        5. Train for N epochs
        6. Save model + tokenizer + config
    
    Returns:
        model, tokenizer, output_dir
    """
    # fix all random seeds for reproducibility
    # same seed = same results every run
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # use GPU if available, otherwise CPU
    # for this small model CPU is fast enough (~0.3s per epoch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Step 1: Load data ──────────────────────────────────────
    train_data, val_data, test_data = load_dataset()

    # override batch size to match actual number of intents loaded
    # this guarantees one pair per intent per batch → zero false negatives
    # works for both 20 intents (synthetic) and 150 intents (CLINC)
    n_intents = len(train_data)
    if args.batch_size != n_intents:
        print(f"\n  Adjusting batch size: {args.batch_size} → {n_intents} (= n_intents)")
        args.batch_size = n_intents

    # ── Step 2: Build tokenizer ────────────────────────────────
    print("\nBuilding tokenizer...")

    # flatten all training texts into one list for fitting
    # e.g. [q for intent 0] + [q for intent 1] + ... = 3000 texts
    all_texts = [
        text
        for queries in train_data.values()
        for text in queries
    ]

    tokenizer = SimpleTokenizer(max_vocab_size=8000, max_seq_len=32)
    tokenizer.fit(all_texts)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Step 3: Build model ────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(tokenizer)
    model = model.to(device)
    print(f"  Parameters: {model.n_parameters():,}")

    # ── Step 4: Training pairs + DataLoader ────────────────────
    print("\nCreating training pairs...")
    pairs, intent_ids = create_training_pairs(
        train_data,
        pairs_per_intent=args.pairs_per_intent,
        seed=args.seed,
    )

    # custom sampler → one pair per intent per batch
    # → zero false negatives guaranteed
    sampler = IntentAwareBatchSampler(
        intent_ids,
        batch_size=args.batch_size,
    )

    dataset    = PairDataset(pairs)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=make_collate_fn(tokenizer),
    )

    print(f"  Pairs         : {len(pairs):,}")
    print(f"  Batches/epoch : {len(sampler)}")

    # ── Step 5: Optimizer + scheduler + loss ───────────────────
    # AdamW = Adam with weight decay
    # weight decay = small penalty on large weights → prevents overfitting
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )

    # total steps = batches per epoch × number of epochs
    # e.g. 20 batches × 15 epochs = 300 total steps
    total_steps = len(sampler) * args.epochs
    scheduler   = make_scheduler(optimizer, args.warmup_steps, total_steps)
    loss_fn     = MultipleNegativesRankingLoss(temperature=args.temperature)

    # ── Step 6: Training loop ──────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    history = []

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            epoch,
        )
        # store loss and accuracy for each epoch
        history.append(metrics)

    # ── Step 7: Save artifacts ─────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # save model weights — state_dict contains all learned parameters
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # save tokenizer vocabulary — needed at inference time
    tok_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tok_path)

    # save training config + loss/accuracy history
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "epochs":           args.epochs,
            "batch_size":       args.batch_size,
            "lr":               args.lr,
            "temperature":      args.temperature,
            "pairs_per_intent": args.pairs_per_intent,
            "vocab_size":       tokenizer.vocab_size,
            "history":          history,
        }, f, indent=2)

    # generate and save training charts
    plot_training_history(history, args.output_dir)

    print(f"\n  Model     → {model_path}")
    print(f"  Tokenizer → {tok_path}")
    print(f"  Config    → {config_path}")

    return model, tokenizer, args.output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniIntentEmbedder")

    # number of full passes through the training data
    # more epochs = more learning but risk of overfitting
    parser.add_argument("--epochs",           type=int,   default=15)

    # number of pairs per batch
    # automatically overridden to match n_intents after data is loaded
    # so zero false negatives are always guaranteed regardless of dataset
    parser.add_argument("--batch-size",       type=int,   default=20)

    # how many training pairs to create per intent
    # 20 intents × 20 pairs = 400 total pairs
    parser.add_argument("--pairs-per-intent", type=int,   default=20)

    # learning rate — how big each weight update step is
    # 3e-4 is a good default for AdamW on small models
    parser.add_argument("--lr",               type=float, default=3e-4)

    # temperature for contrastive loss
    # 0.05 = sharp, hard training signal
    parser.add_argument("--temperature",      type=float, default=0.05)

    # number of warmup steps before full lr is applied
    # 50 steps = ~2-3 epochs of warmup
    parser.add_argument("--warmup-steps",     type=int,   default=50)

    # where to save model, tokenizer, config
    parser.add_argument("--output-dir",       type=str,   default="../models/finetuned")

    # random seed for reproducibility
    parser.add_argument("--seed",             type=int,   default=42)

    args = parser.parse_args()
    train(args)