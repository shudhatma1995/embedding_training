"""
Inference Demo: Customer Intent Matching (Before vs After Fine-tuning)
======================================================================
Shows how the fine-tuned model handles real-world customer support queries
compared to the classical TF-IDF baseline.

Demonstrates:
  1. Top-3 intent retrieval for curated queries
  2. Semantic similarity shift: same-intent vs cross-intent pairs
  3. Hard cases where fine-tuning helps most

Usage:
    cd customer_intent_search
    python inference_demo.py [--model-dir ../models/finetuned]
"""
import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np

from data_preparation import load_clinc_dataset, build_retrieval_corpus
from evaluate import embed_with_model, embed_with_tfidf
from model import MiniIntentEmbedder, MiniIntentConfig
from tokenizer import SimpleTokenizer
from synthetic_data import INTENTS, get_intent_name


DEMO_QUERIES = [
    # Paraphrases of simple banking queries
    ("What is my remaining balance?",           "balance"),
    ("How much cash do I have left?",           "balance"),
    ("I need to wire funds to my sister",       "transfer"),
    ("My debit card was stolen",                "report_lost_card"),
    ("Please block my account for security",   "freeze_account"),
    ("Show me what I bought last month",        "transactions"),
    ("What are the fees for using abroad?",     "international_fees"),
    ("My payment was rejected at checkout",     "card_declined"),
    ("How do I improve my credit rating?",      "credit_score"),
    ("I want to pay my electricity bill",       "pay_bill"),
]

SIMILARITY_PAIRS = [
    # Same intent — should have HIGH similarity after fine-tuning
    ("What is my balance?",
     "How much money is in my account?",
     True, "Same intent (balance)"),
    ("I lost my debit card",
     "My bank card has gone missing",
     True, "Same intent (report_lost_card)"),
    ("Transfer $200 to my friend",
     "Send money to someone",
     True, "Same intent (transfer)"),
    # Different intent — should have LOW similarity after fine-tuning
    ("What is my balance?",
     "Book me a flight to Paris",
     False, "Different intents (balance vs book_flight)"),
    ("My card was declined",
     "Good morning, how are you?",
     False, "Different intents (card_declined vs greeting)"),
]


def load_finetuned(model_dir: str):
    """Load the fine-tuned model and tokenizer."""
    model_path = os.path.join(model_dir, "model.pt")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    config_path = os.path.join(model_dir, "training_config.json")

    if not all(os.path.exists(p) for p in [model_path, tokenizer_path, config_path]):
        return None, None

    tokenizer = SimpleTokenizer.load(tokenizer_path)
    with open(config_path) as f:
        cfg = json.load(f)["model_config"]

    import torch
    model = MiniIntentEmbedder(MiniIntentConfig(**cfg))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, tokenizer


def retrieve_top_k_intent_names(
    query: str,
    emb_fn,
    corpus_embs: np.ndarray,
    corpus_labels: List[int],
    corpus_texts: List[str],
    k: int = 3,
) -> List[Tuple[str, float, str]]:
    """Return list of (intent_name, score, example_text) for top-k results."""
    import numpy as np
    q_emb = emb_fn([query])
    sims = np.dot(q_emb, corpus_embs.T)[0]
    top_k = np.argsort(sims)[::-1][:k]
    return [
        (get_intent_name(corpus_labels[i]), float(sims[i]), corpus_texts[i])
        for i in top_k
    ]


def run_demo(model_dir: str):
    """Run the full inference demo."""
    print("=" * 65)
    print("  Customer Intent Matching — Inference Demo")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_data, val_data, test_data = load_clinc_dataset()
    corpus_texts, corpus_labels, _ = build_retrieval_corpus(val_data)
    all_train_texts = [q for qs in train_data.values() for q in qs]

    # ── Load fine-tuned model ─────────────────────────────────────────────────
    ft_model, tokenizer = load_finetuned(model_dir)
    has_finetuned = ft_model is not None

    # ── Build TF-IDF ─────────────────────────────────────────────────────────
    print("\n  Building TF-IDF baseline...")
    corpus_tfidf, vectorizer, svd = embed_with_tfidf(
        corpus_texts + all_train_texts, fit=True, n_components=128
    )
    corpus_tfidf = corpus_tfidf[:len(corpus_texts)]

    def tfidf_embed(texts):
        embs, _, _ = embed_with_tfidf(texts, vectorizer=vectorizer, svd=svd)
        return embs

    # ── Fine-tuned embedding function ──────────────────────────────────────────
    if has_finetuned:
        corpus_ft = embed_with_model(ft_model, tokenizer, corpus_texts)
        def ft_embed(texts):
            return embed_with_model(ft_model, tokenizer, texts)
    else:
        print("  Fine-tuned model not found — showing TF-IDF only")

    # ── Demo 1: Top-3 intent retrieval ───────────────────────────────────────
    print("\n" + "─" * 65)
    print("  DEMO 1: Top-3 Intent Retrieval")
    print("─" * 65)

    correct_tfidf = 0
    correct_ft = 0
    total = len(DEMO_QUERIES)

    for query, expected in DEMO_QUERIES:
        print(f"\n  Query: \"{query}\"")
        print(f"  Expected intent: {expected}")

        # TF-IDF
        tfidf_top = retrieve_top_k_intent_names(
            query, tfidf_embed, corpus_tfidf, corpus_labels, corpus_texts
        )
        top1_tfidf = tfidf_top[0][0]
        mark_tf = "✓" if expected in top1_tfidf else "✗"
        print(f"\n  TF-IDF top-3 {mark_tf}:")
        for name, score, ex in tfidf_top:
            print(f"    [{score:.3f}] {name:<25} → \"{ex[:45]}\"")

        if expected in top1_tfidf:
            correct_tfidf += 1

        if has_finetuned:
            ft_top = retrieve_top_k_intent_names(
                query, ft_embed, corpus_ft, corpus_labels, corpus_texts
            )
            top1_ft = ft_top[0][0]
            mark_ft = "✓" if expected in top1_ft else "✗"
            print(f"\n  Fine-tuned top-3 {mark_ft}:")
            for name, score, ex in ft_top:
                print(f"    [{score:.3f}] {name:<25} → \"{ex[:45]}\"")

            if expected in top1_ft:
                correct_ft += 1

    print(f"\n  {'─'*50}")
    print(f"  Demo Recall@1: TF-IDF = {correct_tfidf}/{total}  |  "
          + (f"Fine-tuned = {correct_ft}/{total}" if has_finetuned else ""))

    # ── Demo 2: Cosine similarity shift ───────────────────────────────────────
    print("\n" + "─" * 65)
    print("  DEMO 2: Semantic Similarity — Same vs Different Intent")
    print("─" * 65)
    print("  (Higher = more similar; fine-tuning should push same-intent")
    print("   pairs together and cross-intent pairs apart)\n")

    for t1, t2, same_intent, label in SIMILARITY_PAIRS:
        e1_tf = tfidf_embed([t1])[0]
        e2_tf = tfidf_embed([t2])[0]
        sim_tf = float(np.dot(e1_tf, e2_tf))

        print(f"  {label}")
        print(f"    A: \"{t1}\"")
        print(f"    B: \"{t2}\"")
        print(f"    TF-IDF cosine: {sim_tf:.3f}")

        if has_finetuned:
            e1_ft = ft_embed([t1])[0]
            e2_ft = ft_embed([t2])[0]
            sim_ft = float(np.dot(e1_ft, e2_ft))
            delta = sim_ft - sim_tf
            direction = "↑" if (same_intent and delta > 0) or (not same_intent and delta < 0) else "↓"
            sign = "+" if delta >= 0 else ""
            print(f"    Fine-tuned:    {sim_ft:.3f}  ({sign}{delta:.3f}) {direction}")
        print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="../models/finetuned")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.model_dir)
