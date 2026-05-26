"""
Utility functions for the customer intent embedding project.
"""
import os
import json
import random
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_results(results: dict, path: str):
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: str) -> dict:
    """Load evaluation results from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compute_recall_at_k(retrieved_labels: list, true_label: str, k: int) -> float:
    """
    Compute Recall@K for a single query.
    Returns 1.0 if true_label appears in the top-k retrieved items, else 0.0.
    """
    return float(true_label in retrieved_labels[:k])


def compute_mrr(retrieved_labels: list, true_label: str, cutoff: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank for a single query.
    Returns the reciprocal of the rank of the first correct result.
    """
    for rank, label in enumerate(retrieved_labels[:cutoff], start=1):
        if label == true_label:
            return 1.0 / rank
    return 0.0


def print_banner(title: str, width: int = 60):
    """Print a formatted section banner."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def format_metric(name: str, value: float) -> str:
    """Format a metric name/value pair for display."""
    return f"  {name:<25} {value:.4f} ({value*100:.2f}%)"


INTENT_DESCRIPTIONS = {
    "transfer": "Move money between accounts or to another person",
    "transactions": "View recent account transactions and history",
    "balance": "Check account balance",
    "freeze_account": "Temporarily freeze/lock an account for security",
    "pay_bill": "Pay a bill or utility",
    "exchange_rate": "Get currency exchange rates",
    "credit_score": "Check or improve credit score",
    "report_lost_card": "Report a lost or stolen card",
    "card_acceptance": "Check where a card is accepted",
    "spending_history": "Review spending patterns and history",
}
