"""
Evaluation: Baseline vs Fine-tuned MiniIntent Embedding Model
==============================================================
Compares three embedding strategies on customer intent retrieval:
  1. Random-init (untrained model — the true zero-shot baseline)
  2. TF-IDF + SVD (classical embedding baseline)
  3. Fine-tuned MiniIntentEmbedder (contrastive learning on intent pairs)

Evaluation Protocol (k-NN Intent Retrieval):
  - Corpus: one representative query per intent (from validation set)
  - Query: each test query is embedded and matched against corpus
  - Metric: is the retrieved intent the correct one?

Metrics:
  - Recall@1    — top-1 result is the correct intent   (= Accuracy)
  - Recall@3    — correct intent in top-3 results
  - Recall@5    — correct intent in top-5 results
  - MRR@5       — Mean Reciprocal Rank at cutoff 5

Usage:
    python evaluate.py [--model-dir ../models/finetuned] [--output-dir ../results]
"""
import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data_preparation import load_clinc_dataset, build_retrieval_corpus
from model import MiniIntentEmbedder, MiniIntentConfig, build_model
from tokenizer import SimpleTokenizer
from utils import compute_recall_at_k, compute_mrr, print_banner, save_results


# ─────────────────────────────────────────────────────────────────────────────
# Embedding methods
# ─────────────────────────────────────────────────────────────────────────────

def embed_with_model(
    model: MiniIntentEmbedder,
    tokenizer: SimpleTokenizer,
    texts: List[str],
    batch_size: int = 128,
) -> np.ndarray:
    """Encode texts using MiniIntentEmbedder."""
    return model.encode(texts, tokenizer, batch_size=batch_size, device="cpu")


def embed_with_tfidf(
    texts: List[str],
    vectorizer=None,
    svd=None,
    fit: bool = False,
    n_components: int = 128,
):
    """
    Embed texts using TF-IDF + Truncated SVD (Latent Semantic Analysis).

    This is the classical embedding baseline — no training, just counting.
    Returns (embeddings, vectorizer, svd).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=10000,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = svd.fit_transform(tfidf_matrix)
    else:
        tfidf_matrix = vectorizer.transform(texts)
        dense = svd.transform(tfidf_matrix)

    embs = normalize(dense, norm="l2", axis=1)
    return embs, vectorizer, svd


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval evaluation
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    corpus_labels: List[int],
    k: int = 5,
) -> List[List[int]]:
    """
    For each query, retrieve top-k corpus items by cosine similarity.
    Both embeddings should be L2-normalised (dot product == cosine sim).
    """
    sim = np.dot(query_embs, corpus_embs.T)   # (n_queries, n_corpus)
    top_k = np.argsort(sim, axis=1)[:, ::-1][:, :k]
    return [[corpus_labels[idx] for idx in row] for row in top_k]


def compute_metrics(
    retrieved: List[List[int]],
    true_labels: List[int],
) -> Dict:
    """Compute all retrieval metrics over the full test set."""
    r1, r3, r5, mrr = [], [], [], []

    for true, ranked in zip(true_labels, retrieved):
        r1.append(compute_recall_at_k(ranked, true, k=1))
        r3.append(compute_recall_at_k(ranked, true, k=3))
        r5.append(compute_recall_at_k(ranked, true, k=5))
        mrr.append(compute_mrr(ranked, true, cutoff=5))

    # Per-intent Recall@1
    per_intent: Dict[int, List[float]] = {}
    for true, r in zip(true_labels, r1):
        per_intent.setdefault(true, []).append(r)
    intent_scores = {k: float(np.mean(v)) for k, v in per_intent.items()}
    sorted_by_score = sorted(intent_scores.items(), key=lambda x: x[1])

    return {
        "recall@1": float(np.mean(r1)),
        "recall@3": float(np.mean(r3)),
        "recall@5": float(np.mean(r5)),
        "mrr@5": float(np.mean(mrr)),
        "per_intent_recall1": intent_scores,
        "worst_5_intents": sorted_by_score[:5],
        "best_5_intents": sorted_by_score[-5:],
        "n_queries": len(true_labels),
    }


def evaluate_method(
    name: str,
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    corpus_labels: List[int],
    query_labels: List[int],
) -> Dict:
    """Evaluate one embedding method and return its metric dict."""
    retrieved = retrieve_top_k(query_embs, corpus_embs, corpus_labels, k=5)
    metrics = compute_metrics(retrieved, query_labels)
    metrics["model"] = name
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: Dict, output_dir: str):
    """Generate and save comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    names = list(results.keys())
    colors = ["#4878D0", "#6ACC65", "#EE854A"][:len(names)]
    metrics = ["recall@1", "recall@3", "recall@5", "mrr@5"]
    metric_labels = ["Recall@1", "Recall@3", "Recall@5", "MRR@5"]

    # ── Plot 1: grouped bar chart ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(metrics))
    width = 0.25

    ax = axes[0]
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [results[name][m] * 100 for m in metrics]
        offset = (i - len(names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Intent Retrieval: All Methods Compared", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 115)

    # ── Plot 2: improvement bars ──────────────────────────────────────────────
    if "tfidf" in results and "finetuned" in results:
        ax2 = axes[1]
        baseline_key = "tfidf"
        improvements = [
            (results["finetuned"][m] - results[baseline_key][m]) * 100
            for m in metrics
        ]
        colors_imp = ["#2ca02c" if v >= 0 else "#d62728" for v in improvements]
        bars3 = ax2.bar(metric_labels, improvements, color=colors_imp, alpha=0.85)
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax2.set_title("Fine-tuned vs TF-IDF: Improvement", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Improvement (percentage points)", fontsize=12)
        for bar, val in zip(bars3, improvements):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.3 if val >= 0 else -1.2),
                     f"{val:+.2f}pp", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 2: training loss curve ───────────────────────────────────────────
    config_path = os.path.join(
        os.path.dirname(output_dir), "models", "finetuned", "training_config.json"
    )
    # Also try relative path
    for candidate in [
        config_path,
        "../models/finetuned/training_config.json",
        "../../models/finetuned/training_config.json",
    ]:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            with open(candidate) as f:
                config = json.load(f)
            history = config.get("training_history", [])
            if history:
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                epochs = [h["epoch"] for h in history]
                losses = [h["loss"] for h in history]
                accs = [h["acc"] for h in history]

                ax[0].plot(epochs, losses, "o-", color="#EE854A", linewidth=2)
                ax[0].set_xlabel("Epoch")
                ax[0].set_ylabel("MNR Loss")
                ax[0].set_title("Training Loss", fontweight="bold")

                ax[1].plot(epochs, [a * 100 for a in accs], "o-", color="#4878D0", linewidth=2)
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel("In-batch Accuracy (%)")
                ax[1].set_title("Training Accuracy", fontweight="bold")

                plt.tight_layout()
                curve_path = os.path.join(output_dir, "training_curves.png")
                plt.savefig(curve_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Saved: {curve_path}")
            break

    # ── Plot 3: per-intent comparison ─────────────────────────────────────────
    if "tfidf" in results and "finetuned" in results:
        tfidf_per = results["tfidf"]["per_intent_recall1"]
        ft_per = results["finetuned"]["per_intent_recall1"]
        common = sorted(set(tfidf_per.keys()) & set(ft_per.keys()))

        diff = [ft_per[i] - tfidf_per[i] for i in common]
        colors_per = ["#2ca02c" if d > 0 else ("#d62728" if d < 0 else "#aaaaaa") for d in diff]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(diff)), diff, color=colors_per, alpha=0.75, width=0.8)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xlabel("Intent index")
        ax.set_ylabel("Recall@1 change (fine-tuned − TF-IDF)")
        ax.set_title("Per-Intent Recall@1 Improvement", fontsize=14, fontweight="bold")

        n_imp = sum(1 for d in diff if d > 0)
        n_deg = sum(1 for d in diff if d < 0)
        ax.text(0.02, 0.97, f"Improved: {n_imp} | Degraded: {n_deg} | Unchanged: {len(diff)-n_imp-n_deg}",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        per_path = os.path.join(output_dir, "per_intent_improvement.png")
        plt.savefig(per_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {per_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation routine
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(args):
    """Full evaluation pipeline."""

    # ── Load data ─────────────────────────────────────────────────────────────
    print_banner("Loading Dataset")
    train_data, val_data, test_data = load_clinc_dataset()

    # Retrieval corpus: one example per intent from val set
    corpus_texts, corpus_labels, _ = build_retrieval_corpus(val_data)

    # Flatten test queries
    query_texts, query_labels = [], []
    for intent_id, queries in test_data.items():
        for q in queries:
            query_texts.append(q)
            query_labels.append(intent_id)

    print(f"  Corpus size:   {len(corpus_texts)} (one per intent)")
    print(f"  Test queries:  {len(query_texts):,}")

    all_results = {}

    # ── Method 1: TF-IDF + SVD baseline ──────────────────────────────────────
    print_banner("Method 1: TF-IDF + SVD (LSA) Baseline")
    all_train_texts = [q for qs in train_data.values() for q in qs]

    t0 = time.time()
    corpus_tfidf, vectorizer, svd = embed_with_tfidf(
        corpus_texts + all_train_texts, fit=True, n_components=128
    )
    corpus_tfidf = corpus_tfidf[:len(corpus_texts)]  # keep only corpus portion
    query_tfidf, _, _ = embed_with_tfidf(query_texts, vectorizer=vectorizer, svd=svd)
    tfidf_time = time.time() - t0

    tfidf_results = evaluate_method(
        "TF-IDF + SVD (LSA)",
        query_tfidf, corpus_tfidf, corpus_labels, query_labels
    )
    tfidf_results["encode_time_s"] = round(tfidf_time, 2)
    all_results["tfidf"] = tfidf_results
    print(f"  Recall@1: {tfidf_results['recall@1']:.2%}  "
          f"MRR@5: {tfidf_results['mrr@5']:.2%}  "
          f"[{tfidf_time:.2f}s]")

    # ── Method 2: Random-init model (untrained baseline) ──────────────────────
    print_banner("Method 2: Random-Init Transformer (Untrained)")
    if os.path.isdir(args.model_dir):
        # Load tokenizer and build same architecture (random weights)
        tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
        config_path = os.path.join(args.model_dir, "training_config.json")

        if os.path.exists(tokenizer_path) and os.path.exists(config_path):
            tokenizer = SimpleTokenizer.load(tokenizer_path)
            with open(config_path) as f:
                cfg = json.load(f)["model_config"]

            config = MiniIntentConfig(**cfg)
            random_model = MiniIntentEmbedder(config)  # random weights, NOT fine-tuned

            t0 = time.time()
            corpus_rand = embed_with_model(random_model, tokenizer, corpus_texts)
            query_rand = embed_with_model(random_model, tokenizer, query_texts)
            rand_time = time.time() - t0

            rand_results = evaluate_method(
                "Random-Init Transformer",
                query_rand, corpus_rand, corpus_labels, query_labels
            )
            rand_results["encode_time_s"] = round(rand_time, 2)
            all_results["random"] = rand_results
            print(f"  Recall@1: {rand_results['recall@1']:.2%}  "
                  f"MRR@5: {rand_results['mrr@5']:.2%}  "
                  f"[{rand_time:.2f}s]")
        else:
            print("  Tokenizer/config not found — skipping random baseline")
    else:
        print("  Model dir not found — skipping random baseline")

    # ── Method 3: Fine-tuned model ────────────────────────────────────────────
    print_banner("Method 3: Fine-tuned MiniIntent Transformer")
    if os.path.isdir(args.model_dir):
        model_path = os.path.join(args.model_dir, "model.pt")
        tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
        config_path = os.path.join(args.model_dir, "training_config.json")

        if all(os.path.exists(p) for p in [model_path, tokenizer_path, config_path]):
            tokenizer = SimpleTokenizer.load(tokenizer_path)
            with open(config_path) as f:
                cfg = json.load(f)["model_config"]

            config = MiniIntentConfig(**cfg)
            model = MiniIntentEmbedder(config)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            t0 = time.time()
            corpus_ft = embed_with_model(model, tokenizer, corpus_texts)
            query_ft = embed_with_model(model, tokenizer, query_texts)
            ft_time = time.time() - t0

            ft_results = evaluate_method(
                "Fine-tuned MiniIntent Transformer",
                query_ft, corpus_ft, corpus_labels, query_labels
            )
            ft_results["encode_time_s"] = round(ft_time, 2)
            all_results["finetuned"] = ft_results
            print(f"  Recall@1: {ft_results['recall@1']:.2%}  "
                  f"MRR@5: {ft_results['mrr@5']:.2%}  "
                  f"[{ft_time:.2f}s]")
        else:
            print(f"  Model files not found at {args.model_dir} — run train.py first")
    else:
        print(f"  {args.model_dir} not found — run train.py first")

    # ── Print comparison table ────────────────────────────────────────────────
    print_banner("Results Summary")
    metrics = ["recall@1", "recall@3", "recall@5", "mrr@5"]
    header = f"  {'Method':<35} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'MRR@5':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    method_display = {
        "tfidf": "TF-IDF + SVD (LSA)",
        "random": "Random-Init Transformer",
        "finetuned": "Fine-tuned MiniIntent",
    }
    for key in ["tfidf", "random", "finetuned"]:
        if key not in all_results:
            continue
        res = all_results[key]
        name = method_display[key]
        vals = "  ".join(f"{res[m]*100:>6.2f}%" for m in metrics)
        print(f"  {name:<35} {vals}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print_banner("Generating Plots")
    plot_results(all_results, args.output_dir)

    # ── Save ──────────────────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    save_results(all_results, results_path)

    return all_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="../models/finetuned")
    parser.add_argument("--output-dir", type=str, default="../results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
