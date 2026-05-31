import random
from collections import defaultdict
from typing import Tuple, List, Dict

# InputExample is a simple container class from the sentence-transformers library. It holds a pair of texts that belong together:
from sentence_transformers import InputExample

# InputExample is essentially just this:
# class InputExample:
#     def __init__(self, texts):
#         self.texts = texts   # ["anchor", "positive"]

def load_dataset() -> Tuple[Dict, Dict, Dict]:
    """
    Load e-commerce intent dataset.
    Tries HuggingFace Hub first, falls back to local synthetic data.
    
    Returns:
        train_data, val_data, test_data
        each is a dict of {intent_id: [list of query strings]}
    """
    print("Loading e-commerce intent dataset...")

    try:
        # ── attempt to load real dataset from HuggingFace ──────────
        # requires internet access and the `datasets` package
        from datasets import load_dataset as hf_load

        print("  Trying HuggingFace Hub...")
        dataset = hf_load("clinc/clinc_oos", "small")

        def group_by_intent(split) -> Dict:
            # group all queries by their intent ID
            groups = defaultdict(list)
            for example in split:
                intent_id = example["intent"]
                text = example["text"]
                if intent_id != 42:   # 42 = out-of-scope label, skip it
                    groups[intent_id].append(text)
            return dict(groups)

        train_data = group_by_intent(dataset["train"])
        val_data   = group_by_intent(dataset["validation"])
        test_data  = group_by_intent(dataset["test"])
        source = "HuggingFace Hub (clinc/clinc_oos)"

    except Exception as e:
        # ── fallback to local synthetic e-commerce data ─────────────
        # triggers when: no internet, datasets not installed, etc.
        print(f"  Hub unavailable ({type(e).__name__}). Using synthetic data.")
        from synthetic_data import generate_synthetic_dataset
        train_data, val_data, test_data = generate_synthetic_dataset(
            n_train=150,
            n_val=30,
            n_test=30,
        )
        source = "Synthetic (local e-commerce fallback)"

    # ── print summary ───────────────────────────────────────────────
    n_intents = len(train_data)
    n_train   = sum(len(v) for v in train_data.values())
    n_val     = sum(len(v) for v in val_data.values())
    n_test    = sum(len(v) for v in test_data.values())

    print(f"  Source  : {source}")
    print(f"  Intents : {n_intents}")
    print(f"  Train   : {n_train:,} queries")
    print(f"  Val     : {n_val:,} queries")
    print(f"  Test    : {n_test:,} queries")

    return train_data, val_data, test_data


def create_training_pairs(
    intent_groups: Dict,
    pairs_per_intent: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[InputExample], List[int]]:
    """
    Create positive training pairs for contrastive learning.
    Returns pairs AND their intent IDs so the custom batch
    sampler can guarantee no same-intent pairs in any batch.

    Args:
        intent_groups:    dict of {intent_id: [query texts]}
        pairs_per_intent: how many pairs to sample per intent
        seed:             random seed for reproducibility

    Returns:
        all_pairs      : List of InputExample(texts=[anchor, positive])
        all_intent_ids : List of intent IDs, one per pair (same index)
    """
    random.seed(seed)

    # will hold pairs and their intent labels in sync
    # all_pairs[i] belongs to all_intent_ids[i]
    all_pairs      = []
    all_intent_ids = []

    for intent_id, queries in intent_groups.items():

        # need at least 2 queries to form a pair
        if len(queries) < 2:
            continue

        # shuffle so we pick different pairs each run
        shuffled = queries.copy()
        random.shuffle(shuffled)

        # dont request more pairs than queries allow
        # e.g. 150 queries → min(20, 75) = 20 pairs
        n_pairs = min(pairs_per_intent, len(queries) // 2)

        # take consecutive pairs from shuffled list
        # i=0 → (shuffled[0], shuffled[1])
        # i=2 → (shuffled[2], shuffled[3])
        for i in range(0, n_pairs * 2, 2):
            if i + 1 < len(shuffled):
                all_pairs.append(
                    InputExample(texts=[shuffled[i], shuffled[i + 1]])
                )
                # store intent label alongside the pair
                all_intent_ids.append(intent_id)

    if verbose:
        print(f"  Created {len(all_pairs):,} training pairs "
              f"across {len(intent_groups)} intents")

    return all_pairs, all_intent_ids


class IntentAwareBatchSampler:
    """
    Custom batch sampler that guarantees no two pairs in the same
    batch share the same intent — eliminating false negatives entirely.

    How it works:
        1. Groups pair indices by intent ID
        2. Each batch is built by picking exactly one pair per intent
        3. Cycles through rounds until all pairs are exhausted

    Example with 3 intents, 3 pairs each, batch_size=3:
        round 0 → [intent0_pair0, intent1_pair0, intent2_pair0]
        round 1 → [intent0_pair1, intent1_pair1, intent2_pair1]
        round 2 → [intent0_pair2, intent1_pair2, intent2_pair2]
    """

    def __init__(self, intent_ids: List[int], batch_size: int):
        """
        Args:
            intent_ids : list of intent IDs, one per pair (from create_training_pairs)
            batch_size : must be >= number of unique intents
        """
        self.batch_size = batch_size

        # group pair indices by intent
        # e.g. {0: [0, 20, 40], 1: [1, 21, 41], ...}
        self.intent_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, intent_id in enumerate(intent_ids):
            self.intent_to_indices[intent_id].append(idx)

        self.n_intents = len(self.intent_to_indices)

        # batch_size should be a multiple of n_intents for clean batches
        if batch_size % self.n_intents != 0:
            print(f"  Warning: batch_size {batch_size} is not a multiple of "
                  f"n_intents {self.n_intents}. "
                  f"Some batches may have slight intent imbalance.")

    def __iter__(self):
        """
        Yields one batch at a time.
        Each batch contains pairs from different intents only.
        """
        # shuffle indices within each intent so pairs vary across epochs
        # e.g. intent 0 indices: [0, 20, 40] → [40, 0, 20]
        intent_indices = {}
        for intent_id, indices in self.intent_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            intent_indices[intent_id] = shuffled

        # how many rounds we need — determined by longest intent list
        # e.g. if every intent has 20 pairs → 20 rounds
        max_pairs = max(len(v) for v in intent_indices.values())

        # build a flat ordered list: one pair per intent per round
        # round 0: [intent0[0], intent1[0], intent2[0], ...]
        # round 1: [intent0[1], intent1[1], intent2[1], ...]
        ordered_indices = []
        for round_idx in range(max_pairs):
            for intent_id, indices in intent_indices.items():
                # some intents may run out before others — skip them
                if round_idx < len(indices):
                    ordered_indices.append(indices[round_idx])

        # slice ordered list into batches and yield each batch
        # e.g. batch_size=20: [0:20], [20:40], [40:60], ...
        for start in range(0, len(ordered_indices), self.batch_size):
            batch = ordered_indices[start: start + self.batch_size]
            # skip incomplete final batch to keep loss stable
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        """Total number of complete batches per epoch."""
        total_pairs = sum(len(v) for v in self.intent_to_indices.values())
        return total_pairs // self.batch_size


def build_retrieval_corpus(intent_groups: Dict) -> Tuple[List, List, Dict]:
    """
    Build a retrieval corpus for evaluation.

    One representative query per intent forms the corpus.
    Remaining queries become evaluation queries.

    Returns:
        corpus_texts  : list of representative queries (one per intent)
        corpus_labels : corresponding intent IDs
        eval_queries  : dict of {intent_id: [remaining queries]}
    """
    corpus_texts  = []
    corpus_labels = []
    eval_queries  = {}

    for intent_id, queries in intent_groups.items():
        # first query = corpus representative
        corpus_texts.append(queries[0])
        corpus_labels.append(intent_id)
        # rest = evaluation queries
        eval_queries[intent_id] = queries[1:]

    return corpus_texts, corpus_labels, eval_queries