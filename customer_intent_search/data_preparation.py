"""
Data Preparation for Customer Intent Embedding Training
========================================================
Loads the CLINC OOS dataset from HuggingFace when internet is available,
or falls back to a locally generated synthetic dataset with the same structure.

Dataset: clinc/clinc_oos  (HuggingFace)
  - 150 in-scope intent classes across 10 domains
  - 22,500 training queries (~150 per intent)
  - 4,500 test queries (~30 per intent)
  - Plus 1,200 out-of-scope queries

Synthetic fallback:
  - 20 intent classes across 5 banking/fintech domains
  - Same structure and API — just smaller scale

HuggingFace page: https://huggingface.co/datasets/clinc/clinc_oos
Paper: https://arxiv.org/abs/1909.02027
"""
import random
from collections import defaultdict
from typing import Tuple

from sentence_transformers import InputExample


def _load_clinc_from_hub() -> Tuple[dict, dict, dict]:
    """
    Attempt to load the real CLINC OOS dataset from HuggingFace Hub.

    Raises ImportError or FileNotFoundError if Hub is unreachable.
    """
    from datasets import load_dataset

    print("  Attempting to load clinc/clinc_oos from HuggingFace Hub...")
    dataset = load_dataset("clinc/clinc_oos", "small")

    def group_by_intent(split) -> dict:
        groups = defaultdict(list)
        for example in split:
            intent_id = example["intent"]
            text = example["text"]
            if intent_id != 42:          # 42 = out-of-scope label
                groups[intent_id].append(text)
        return dict(groups)

    train_data = group_by_intent(dataset["train"])
    val_data = group_by_intent(dataset["validation"])
    test_data = group_by_intent(dataset["test"])
    return train_data, val_data, test_data


def load_clinc_dataset(split_type: str = "small") -> Tuple[dict, dict, dict]:
    """
    Load the CLINC OOS (or synthetic equivalent) intent dataset.

    Tries HuggingFace Hub first; falls back to the locally generated
    synthetic dataset if the Hub is unreachable.

    Args:
        split_type: ignored for synthetic data; passed to Hub loader

    Returns:
        train_data, val_data, test_data — each a dict of {intent_id: [texts]}
    """
    print("Loading customer intent dataset...")

    try:
        train_data, val_data, test_data = _load_clinc_from_hub()
        source = "HuggingFace Hub (clinc/clinc_oos)"
    except Exception as e:
        print(f"  Hub unavailable ({type(e).__name__}). "
              f"Using local synthetic dataset.")
        print("  (Run with internet access to use the full CLINC OOS dataset)")
        from synthetic_data import generate_synthetic_dataset
        train_data, val_data, test_data = generate_synthetic_dataset(
            n_train=150, n_val=30, n_test=30
        )
        source = "Synthetic (local fallback for CLINC OOS)"

    n_intents = len(train_data)
    n_train = sum(len(v) for v in train_data.values())
    n_val = sum(len(v) for v in val_data.values())
    n_test = sum(len(v) for v in test_data.values())

    print(f"  Source:             {source}")
    print(f"  Intents (in-scope): {n_intents}")
    print(f"  Train queries:      {n_train:,}")
    print(f"  Val queries:        {n_val:,}")
    print(f"  Test queries:       {n_test:,}")

    return train_data, val_data, test_data


def create_training_pairs(
    intent_groups: dict,
    pairs_per_intent: int = 20,
    seed: int = 42,
) -> list:
    """
    Create positive training pairs (same intent = similar semantics).

    Strategy: For each intent, randomly sample pairs of queries.
    These pairs are used with MultipleNegativesRankingLoss where
    all other pairs in the batch act as implicit negatives.

    With a batch size of B, each positive pair has B-1 in-batch negatives,
    making this highly efficient for contrastive learning.

    Args:
        intent_groups: dict of {intent_id: [query_texts]}
        pairs_per_intent: number of positive pairs to sample per intent
        seed: random seed

    Returns:
        List of InputExample objects (anchor, positive)
    """
    random.seed(seed)
    pairs = []

    for intent_id, queries in intent_groups.items():
        if len(queries) < 2:
            continue
        n_pairs = min(pairs_per_intent, len(queries) // 2)
        shuffled = queries.copy()
        random.shuffle(shuffled)

        for i in range(0, n_pairs * 2, 2):
            if i + 1 < len(shuffled):
                pairs.append(
                    InputExample(texts=[shuffled[i], shuffled[i + 1]])
                )

    random.shuffle(pairs)
    print(f"  Created {len(pairs):,} positive training pairs "
          f"across {len(intent_groups)} intents")
    return pairs


def build_retrieval_corpus(intent_groups: dict) -> Tuple[list, list, dict]:
    """
    Build a retrieval corpus for evaluation.

    One representative query per intent forms the corpus.
    The remaining queries become evaluation queries.

    Returns:
        corpus_texts: list of representative queries (one per intent)
        corpus_labels: corresponding intent IDs
        eval_queries: dict of {intent_id: [remaining queries]}
    """
    corpus_texts = []
    corpus_labels = []
    eval_queries = {}

    for intent_id, queries in intent_groups.items():
        corpus_texts.append(queries[0])
        corpus_labels.append(intent_id)
        eval_queries[intent_id] = queries[1:]

    return corpus_texts, corpus_labels, eval_queries


def get_dataset_info() -> dict:
    """Return metadata about the CLINC OOS dataset."""
    return {
        "name": "clinc/clinc_oos",
        "url": "https://huggingface.co/datasets/clinc/clinc_oos",
        "paper": "https://arxiv.org/abs/1909.02027",
        "citation": (
            "@inproceedings{larson-etal-2019-evaluation,"
            " title={An Evaluation Dataset for Intent Classification and "
            "Out-of-Scope Prediction},"
            " author={Larson, Stefan and Mahendran, Anish and Peper, Joseph J "
            "and Clarke, Christopher and Lee, Andrew and Hill, Parker and "
            "Kummerfeld, Jonathan K and Leach, Kevin and Laurenzano, Michael A "
            "and Tang, Lingjia and Mars, Jason},"
            " booktitle={Proceedings of EMNLP-IJCNLP 2019}}"
        ),
        "description": (
            "CLINC Out-of-Scope (OOS) Intent Classification dataset. "
            "150 in-scope intent classes across 10 domains: banking, credit_cards, "
            "duplication_and_password, home, auto_and_commute, travel, "
            "utility, work, small_talk, meta. Contains 22,500 training examples "
            "and 4,500 test examples plus 1,200 out-of-scope queries."
        ),
        "domains": [
            "banking", "credit_cards", "duplication_and_password",
            "home", "auto_and_commute", "travel", "utility",
            "work", "small_talk", "meta"
        ],
        "n_intents": 150,
        "splits": {
            "train": 15000,
            "validation": 3000,
            "test": 4500,
            "oos_train": 100,
            "oos_val": 100,
            "oos_test": 1000,
        },
    }


if __name__ == "__main__":
    train_data, val_data, test_data = load_clinc_dataset()

    print("\nSample intents and queries:")
    for intent_id, queries in list(train_data.items())[:3]:
        print(f"\n  Intent {intent_id}:")
        for q in queries[:3]:
            print(f"    - {q}")

    pairs = create_training_pairs(train_data, pairs_per_intent=20)
    print(f"\nSample training pair:")
    print(f"  Anchor:   {pairs[0].texts[0]}")
    print(f"  Positive: {pairs[0].texts[1]}")
