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

from tokenizer import SimpleTokenizer
from model import build_model
from data_preparation import load_dataset, create_training_pairs, IntentAwareBatchSampler


# ── Loss Function ──────────────────────────────────────────────────────────────

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


# ── Dataset ────────────────────────────────────────────────────────────────────

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
        return len(self.pairs)

    def __getitem__(self, idx):
        # returns (anchor_text, positive_text) for this index
        anchor   = self.pairs[idx].texts[0]
        positive = self.pairs[idx].texts[1]
        return anchor, positive


# ── Collate Function ───────────────────────────────────────────────────────────

def make_collate_fn(tokenizer):
    """
    Returns a closure that tokenizes a batch of (anchor, positive) pairs.
    DataLoader calls this to convert raw strings into tensors.
    """

    def collate_fn(batch):
        # separate anchors and positives from list of tuples
        anchors   = [item[0] for item in batch]
        positives = [item[1] for item in batch]

        # tokenize both into tensors of shape (batch_size, max_seq_len)
        anchor_ids,   anchor_mask   = tokenizer.encode_batch(anchors)
        positive_ids, positive_mask = tokenizer.encode_batch(positives)

        return anchor_ids, anchor_mask, positive_ids, positive_mask

    return collate_fn


# ── Learning Rate Scheduler ────────────────────────────────────────────────────

def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup then linear decay to 10% of initial LR.

    warmup: steps 0 → warmup_steps  — lr ramps 0 → initial_lr
    decay:  steps warmup_steps → total_steps — lr ramps initial_lr → 0.1*initial_lr
    """

    def lr_lambda(current_step: int):
        # warmup phase: ramp from 0.0 → 1.0
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # decay phase: ramp from 1.0 → 0.1
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 1.0 - 0.9 * progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Validation Evaluation ──────────────────────────────────────────────────────

def evaluate_val(
    model,
    val_data: dict,
    tokenizer,
    loss_fn,
    device: str,
    pairs_per_intent: int = 10,
    seed: int = 42,
) -> dict:
    """
    Evaluate the model on the validation set.

    Why we need this:
        Training metrics only tell us how well the model fits training data.
        Validation metrics tell us how well it generalises to unseen data.
        If train loss drops but val loss rises → overfitting.

    How it works:
        1. Create val pairs from val_data (separate from training pairs)
        2. Run model in eval mode (dropout disabled)
        3. Compute val loss and val Recall@1 accuracy
        4. No gradient updates — purely measurement

    Returns:
        dict with val_loss and val_accuracy
    """

    # create validation pairs — different queries from the val split
    # uses a different seed from training to avoid overlap
    val_pairs, val_intent_ids = create_training_pairs(
        val_data,
        pairs_per_intent=pairs_per_intent,
        seed=seed + 1,   # different seed from training pairs
    )

    # build val dataloader — same structure as training
    n_intents  = len(val_data)
    val_sampler = IntentAwareBatchSampler(val_intent_ids, batch_size=n_intents)
    val_dataset = PairDataset(val_pairs)
    val_loader  = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=make_collate_fn(tokenizer),
    )

    # switch to eval mode — disables dropout so results are deterministic
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_pairs   = 0

    # torch.no_grad() — skip gradient computation entirely
    # saves memory and speeds up inference
    with torch.no_grad():
        for anchor_ids, anchor_mask, positive_ids, positive_mask in val_loader:

            anchor_ids    = anchor_ids.to(device)
            anchor_mask   = anchor_mask.to(device)
            positive_ids  = positive_ids.to(device)
            positive_mask = positive_mask.to(device)

            # forward pass — same as training but no backward
            anchor_emb   = model(anchor_ids,   anchor_mask)
            positive_emb = model(positive_ids, positive_mask)

            # compute val loss
            loss = loss_fn(anchor_emb, positive_emb)
            total_loss += loss.item()

            # compute val accuracy — argmax of similarity matrix diagonal
            sim       = torch.matmul(anchor_emb, positive_emb.T)
            predicted = sim.argmax(dim=1)
            labels    = torch.arange(sim.shape[0], device=device)
            total_correct += (predicted == labels).sum().item()
            total_pairs   += sim.shape[0]

    # switch back to training mode for next epoch
    model.train()

    return {
        "val_loss":     total_loss / len(val_loader),
        "val_accuracy": total_correct / total_pairs,
    }


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device: str,
    epoch: int,
    val_data: dict = None,
    tokenizer=None,
) -> dict:
    """
    Run one full pass over all training batches.
    Optionally evaluates on val set at end of epoch.

    Returns dict with train_loss, train_accuracy, and optionally val metrics.
    """

    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_pairs   = 0
    start_time    = time.time()

    for anchor_ids, anchor_mask, positive_ids, positive_mask in dataloader:

        # move to device
        anchor_ids    = anchor_ids.to(device)
        anchor_mask   = anchor_mask.to(device)
        positive_ids  = positive_ids.to(device)
        positive_mask = positive_mask.to(device)

        # forward pass
        anchor_emb   = model(anchor_ids,   anchor_mask)
        positive_emb = model(positive_ids, positive_mask)

        # compute loss
        loss = loss_fn(anchor_emb, positive_emb)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # track training metrics
        total_loss += loss.item()

        with torch.no_grad():
            sim           = torch.matmul(anchor_emb, positive_emb.T)
            predicted     = sim.argmax(dim=1)
            labels        = torch.arange(sim.shape[0], device=device)
            total_correct += (predicted == labels).sum().item()
            total_pairs   += sim.shape[0]

    elapsed        = time.time() - start_time
    train_loss     = total_loss / len(dataloader)
    train_accuracy = total_correct / total_pairs

    # build metrics dict — start with training metrics
    metrics = {
        "loss":     train_loss,
        "accuracy": train_accuracy,
    }

    # optionally evaluate on validation set
    if val_data is not None and tokenizer is not None:
        val_metrics = evaluate_val(model, val_data, tokenizer, loss_fn, device)
        metrics["val_loss"]     = val_metrics["val_loss"]
        metrics["val_accuracy"] = val_metrics["val_accuracy"]

        # overfitting indicator: gap between train and val loss
        # small gap  → generalising well
        # large gap  → memorising training data
        gap = val_metrics["val_loss"] - train_loss
        overfit_flag = " ⚠ overfit?" if gap > 1.0 else ""

        print(f"  Epoch {epoch:02d} | "
              f"train_loss {train_loss:.4f} | train_acc {train_accuracy*100:.1f}% | "
              f"val_loss {val_metrics['val_loss']:.4f} | val_acc {val_metrics['val_accuracy']*100:.1f}% | "
              f"time {elapsed:.1f}s{overfit_flag}")
    else:
        print(f"  Epoch {epoch:02d} | "
              f"loss {train_loss:.4f} | acc {train_accuracy*100:.1f}% | "
              f"time {elapsed:.1f}s")

    return metrics


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_training_history(history: list, output_dir: str, label: str = ""):
    """
    Generate and save training charts showing train vs val curves.

    Four subplots:
        Top-left:  Train loss vs Val loss     — overfitting visible as gap
        Top-right: Train acc  vs Val acc      — generalisation visible
        Bottom-left:  Loss gap (val-train)    — direct overfitting signal
        Bottom-right: Val accuracy trend      — best val accuracy highlighted

    If val metrics are not in history, falls back to train-only plots.
    """

    epochs      = list(range(1, len(history) + 1))
    train_loss  = [h["loss"]     for h in history]
    train_acc   = [h["accuracy"] * 100 for h in history]
    has_val     = "val_loss" in history[0]

    if has_val:
        val_loss = [h["val_loss"]     for h in history]
        val_acc  = [h["val_accuracy"] * 100 for h in history]

    title = f"Training History — MiniIntentEmbedder"
    if label:
        title += f" ({label})"

    if has_val:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── top-left: loss curves ──────────────────────────────────
    ax1.plot(epochs, train_loss, color="#e74c3c", linewidth=2,
             marker="o", markersize=4, label="Train loss")
    if has_val:
        ax1.plot(epochs, val_loss, color="#e67e22", linewidth=2,
                 marker="s", markersize=4, linestyle="--", label="Val loss")
        # shade gap between train and val loss to show overfitting
        ax1.fill_between(epochs, train_loss, val_loss,
                         alpha=0.15, color="#e67e22", label="Overfitting gap")
    ax1.fill_between(epochs, train_loss, alpha=0.08, color="#e74c3c")
    ax1.annotate(f"{train_loss[-1]:.4f}", xy=(epochs[-1], train_loss[-1]),
                 xytext=(-35, 8), textcoords="offset points", fontsize=9, color="#e74c3c")
    if has_val:
        ax1.annotate(f"{val_loss[-1]:.4f}", xy=(epochs[-1], val_loss[-1]),
                     xytext=(-35, -15), textcoords="offset points", fontsize=9, color="#e67e22")
    ax1.set_title("Loss: Train vs Validation", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── top-right: accuracy curves ─────────────────────────────
    ax2.plot(epochs, train_acc, color="#2ecc71", linewidth=2,
             marker="o", markersize=4, label="Train acc")
    if has_val:
        ax2.plot(epochs, val_acc, color="#27ae60", linewidth=2,
                 marker="s", markersize=4, linestyle="--", label="Val acc")
        # mark best val accuracy
        best_val_epoch = val_acc.index(max(val_acc)) + 1
        best_val_value = max(val_acc)
        ax2.axvline(x=best_val_epoch, color="gray", linestyle=":", alpha=0.7)
        ax2.annotate(f"best val\n{best_val_value:.1f}%",
                     xy=(best_val_epoch, best_val_value),
                     xytext=(8, -20), textcoords="offset points",
                     fontsize=8, color="#27ae60")
    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.4, label="95% reference")
    ax2.annotate(f"{train_acc[-1]:.1f}%", xy=(epochs[-1], train_acc[-1]),
                 xytext=(-35, 8), textcoords="offset points", fontsize=9, color="#2ecc71")
    ax2.set_title("Recall@1: Train vs Validation", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    if has_val:
        # ── bottom-left: overfitting gap ──────────────────────
        gap = [v - t for v, t in zip(val_loss, train_loss)]
        colors = ["#e74c3c" if g > 0.5 else "#f39c12" if g > 0.2 else "#2ecc71" for g in gap]
        ax3.bar(epochs, gap, color=colors, alpha=0.7)
        ax3.axhline(y=0,   color="black", linewidth=0.8)
        ax3.axhline(y=0.5, color="#e74c3c", linestyle="--", alpha=0.5, label="Overfit threshold")
        ax3.axhline(y=0.2, color="#f39c12", linestyle="--", alpha=0.5, label="Warning threshold")
        ax3.set_title("Overfitting Gap (val_loss − train_loss)", fontsize=12)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss Gap")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")

        # ── bottom-right: val accuracy with trend ─────────────
        ax4.plot(epochs, val_acc, color="#3498db", linewidth=2,
                 marker="o", markersize=4, label="Val accuracy")
        ax4.fill_between(epochs, val_acc, alpha=0.1, color="#3498db")
        # highlight best epoch
        ax4.scatter([best_val_epoch], [best_val_value],
                    color="#e74c3c", s=100, zorder=5, label=f"Best: {best_val_value:.1f}% @ epoch {best_val_epoch}")
        ax4.axhline(y=95, color="gray", linestyle="--", alpha=0.4, label="95% reference")
        ax4.set_title("Validation Accuracy Trend", fontsize=12)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy (%)")
        ax4.set_ylim(0, 108)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # save plot
    fname     = f"training_curves_{label}.png" if label else "training_curves.png"
    plot_path = os.path.join(output_dir, fname)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot      → {plot_path}")
    return plot_path


def plot_experiment_comparison(experiments: list, output_dir: str):
    """
    Compare multiple training runs on the same chart.

    experiments: list of dicts:
        {"label": "temp=0.05", "history": [...], "color": "#e74c3c"}

    Produces two plots:
        Left:  val loss across experiments
        Right: val accuracy across experiments
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Experiment Comparison", fontsize=14, fontweight="bold")

    for exp in experiments:
        history = exp["history"]
        label   = exp["label"]
        color   = exp["color"]
        epochs  = list(range(1, len(history) + 1))

        has_val = "val_loss" in history[0]

        if has_val:
            val_loss = [h["val_loss"]     for h in history]
            val_acc  = [h["val_accuracy"] * 100 for h in history]
            ax1.plot(epochs, val_loss, color=color, linewidth=2,
                     marker="o", markersize=4, label=label)
            ax2.plot(epochs, val_acc,  color=color, linewidth=2,
                     marker="o", markersize=4, label=f"{label} (best: {max(val_acc):.1f}%)")
        else:
            # fallback to train metrics if val not available
            train_loss = [h["loss"]     for h in history]
            train_acc  = [h["accuracy"] * 100 for h in history]
            ax1.plot(epochs, train_loss, color=color, linewidth=2,
                     marker="o", markersize=4, label=label, linestyle="--")
            ax2.plot(epochs, train_acc,  color=color, linewidth=2,
                     marker="o", markersize=4, label=label, linestyle="--")

    ax1.set_title("Validation Loss", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.4, label="95% reference")
    ax2.set_title("Validation Accuracy", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 108)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "experiment_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Comparison plot → {plot_path}")
    return plot_path


# ── Main Train Function ────────────────────────────────────────────────────────

def train(args) -> tuple:
    """
    Full training pipeline with validation tracking.

    Steps:
        1. Load dataset (train + val + test)
        2. Build tokenizer on train data
        3. Build model
        4. Create training pairs + IntentAwareBatchSampler
        5. Train N epochs — evaluate on val set each epoch
        6. Save model, tokenizer, config, plots
    """

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ── Step 1: Load data ──────────────────────────────────────
    train_data, val_data, test_data = load_dataset()

    # auto-set batch size to n_intents for zero false negatives
    n_intents = len(train_data)
    if args.batch_size != n_intents:
        print(f"\n  Adjusting batch size: {args.batch_size} → {n_intents} (= n_intents)")
        args.batch_size = n_intents

    # ── Step 2: Build tokenizer ────────────────────────────────
    print("\nBuilding tokenizer...")
    all_texts = [text for queries in train_data.values() for text in queries]
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

    sampler    = IntentAwareBatchSampler(intent_ids, batch_size=args.batch_size)
    dataset    = PairDataset(pairs)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=make_collate_fn(tokenizer),
    )

    print(f"  Pairs         : {len(pairs):,}")
    print(f"  Batches/epoch : {len(sampler)}")

    # ── Step 5: Optimizer + scheduler + loss ───────────────────
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(sampler) * args.epochs
    scheduler   = make_scheduler(optimizer, args.warmup_steps, total_steps)
    loss_fn     = MultipleNegativesRankingLoss(temperature=args.temperature)

    # ── Step 6: Training loop ──────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  {'Epoch':>5} | {'train_loss':>10} | {'train_acc':>9} | "
          f"{'val_loss':>8} | {'val_acc':>7} | {'time':>6}")
    print("  " + "─" * 65)

    history = []

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model, dataloader, optimizer, scheduler,
            loss_fn, device, epoch,
            val_data=val_data,
            tokenizer=tokenizer,
        )
        history.append(metrics)

    # print overfitting summary
    if "val_loss" in history[-1]:
        final_gap = history[-1]["val_loss"] - history[-1]["loss"]
        best_val  = max(h["val_accuracy"] for h in history) * 100
        best_epoch = max(range(len(history)), key=lambda i: history[i]["val_accuracy"]) + 1
        print(f"\n  Best val accuracy : {best_val:.1f}% at epoch {best_epoch}")
        print(f"  Final loss gap    : {final_gap:.4f} "
              f"({'⚠ overfitting' if final_gap > 0.5 else 'healthy'})")

    # ── Step 7: Save artifacts ─────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    model_path  = os.path.join(args.output_dir, "model.pt")
    tok_path    = os.path.join(args.output_dir, "tokenizer.json")
    config_path = os.path.join(args.output_dir, "config.json")

    torch.save(model.state_dict(), model_path)
    tokenizer.save(tok_path)

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

    # generate plots
    label = getattr(args, "experiment_label", "")
    plot_training_history(history, args.output_dir, label=label)

    print(f"\n  Model     → {model_path}")
    print(f"  Tokenizer → {tok_path}")
    print(f"  Config    → {config_path}")

    return model, tokenizer, args.output_dir


# ── Experiment Runner ──────────────────────────────────────────────────────────

def run_experiments(base_args, results_dir: str):
    """
    Run multiple training experiments with different configs and compare them.

    Experiments:
        1. temperature=0.01  — very sharp signal, very hard negatives
        2. temperature=0.05  — default, sharp signal
        3. temperature=0.20  — soft signal, easier negatives

    Why temperature is interesting:
        - Low temp  → loss is very sensitive to any confusion → fast but risky
        - High temp → loss is forgiving → slower but more stable
        - Finding the sweet spot matters for generalisation

    Saves individual plots per experiment + one comparison plot.
    """

    experiments_config = [
        {"label": "temp=0.01", "temperature": 0.01, "color": "#e74c3c"},
        {"label": "temp=0.05", "temperature": 0.05, "color": "#3498db"},
        {"label": "temp=0.20", "temperature": 0.20, "color": "#2ecc71"},
    ]

    os.makedirs(results_dir, exist_ok=True)
    all_experiments = []

    for config in experiments_config:
        print(f"\n{'='*65}")
        print(f"  EXPERIMENT: {config['label']}")
        print(f"{'='*65}")

        # copy base args and override temperature + output dir
        import copy
        exp_args = copy.deepcopy(base_args)
        exp_args.temperature      = config["temperature"]
        exp_args.experiment_label = config["label"]
        exp_args.output_dir       = os.path.join(results_dir, config["label"].replace("=", "_"))

        _, _, _ = train(exp_args)

        # load history from saved config
        config_path = os.path.join(exp_args.output_dir, "config.json")
        with open(config_path) as f:
            saved = json.load(f)

        all_experiments.append({
            "label":   config["label"],
            "history": saved["history"],
            "color":   config["color"],
        })

    # generate comparison plot
    plot_experiment_comparison(all_experiments, results_dir)

    # print summary table
    print(f"\n{'='*65}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Experiment':<15} | {'Best Val Acc':>12} | {'Best Epoch':>10} | {'Final Val Loss':>14}")
    print("  " + "─" * 58)

    for exp in all_experiments:
        history   = exp["history"]
        has_val   = "val_loss" in history[0]
        if has_val:
            best_acc   = max(h["val_accuracy"] for h in history) * 100
            best_epoch = max(range(len(history)), key=lambda i: history[i]["val_accuracy"]) + 1
            final_loss = history[-1]["val_loss"]
        else:
            best_acc   = max(h["accuracy"] for h in history) * 100
            best_epoch = max(range(len(history)), key=lambda i: history[i]["accuracy"]) + 1
            final_loss = history[-1]["loss"]
        print(f"  {exp['label']:<15} | {best_acc:>11.1f}% | {best_epoch:>10} | {final_loss:>14.4f}")

    # save summary to file
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_experiments, f, indent=2)

    print(f"\n  Summary → {summary_path}")
    return all_experiments


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniIntentEmbedder")

    parser.add_argument("--epochs",           type=int,   default=15)
    parser.add_argument("--batch-size",       type=int,   default=20,
                        help="Auto-overridden to n_intents after data load")
    parser.add_argument("--pairs-per-intent", type=int,   default=20)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--temperature",      type=float, default=0.05)
    parser.add_argument("--warmup-steps",     type=int,   default=50)
    parser.add_argument("--output-dir",       type=str,   default="../models/finetuned")
    parser.add_argument("--results-dir",      type=str,   default="../results")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--experiments",      action="store_true",
                        help="Run temperature comparison experiments")

    args = parser.parse_args()

    if args.experiments:
        # run all temperature experiments and compare
        run_experiments(args, args.results_dir)
    else:
        # single training run
        train(args)
