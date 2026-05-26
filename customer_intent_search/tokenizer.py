"""
Simple Word Tokenizer for Customer Intent Data
===============================================
Builds a vocabulary from training data and provides encoding utilities.
No internet access required — fully self-contained.

Keeps things minimal:
  - Lower-casing + basic punctuation splitting
  - [PAD]=0, [UNK]=1, [CLS]=2 special tokens
  - Vocabulary limited to the most frequent words
"""
import re
import json
import os
from collections import Counter
from typing import List, Dict


SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}


def tokenize(text: str) -> List[str]:
    """Simple word tokenizer: lower-case and split on non-alphanumeric chars."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9']+", text)
    return tokens


class SimpleTokenizer:
    """
    Word-level tokenizer built from a training corpus.

    Vocabulary is the union of the most frequent words across all training
    queries, capped at `max_vocab_size`.
    """

    def __init__(self, max_vocab_size: int = 8000, max_length: int = 32):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, texts: List[str]) -> "SimpleTokenizer":
        """Build vocabulary from a list of texts."""
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        # Start with special tokens
        vocab = dict(SPECIAL_TOKENS)

        # Add most frequent words
        for word, _ in counter.most_common(self.max_vocab_size - len(SPECIAL_TOKENS)):
            if word not in vocab:
                vocab[word] = len(vocab)

        self.word2idx = vocab
        self.idx2word = {idx: word for word, idx in vocab.items()}
        self.vocab_size = len(vocab)
        return self

    def encode(self, text: str) -> List[int]:
        """Encode a single text to a list of token IDs (with [CLS])."""
        tokens = [self.word2idx.get(t, SPECIAL_TOKENS["[UNK]"])
                  for t in tokenize(text)]
        # Prepend [CLS]
        ids = [SPECIAL_TOKENS["[CLS]"]] + tokens
        # Truncate
        ids = ids[:self.max_length]
        return ids

    def encode_batch(
        self, texts: List[str]
    ) -> tuple:
        """
        Encode a batch of texts.

        Returns:
            input_ids: padded token-ID tensor (B, max_length) as list-of-lists
            attention_mask: 1 for real tokens, 0 for padding
        """
        import torch
        encoded = [self.encode(t) for t in texts]
        lengths = [len(e) for e in encoded]
        max_len = min(max(lengths), self.max_length)

        input_ids = []
        attention_mask = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [0] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def save(self, path: str):
        """Save vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "word2idx": self.word2idx,
                "max_vocab_size": self.max_vocab_size,
                "max_length": self.max_length,
                "vocab_size": self.vocab_size,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load a saved tokenizer."""
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(
            max_vocab_size=data["max_vocab_size"],
            max_length=data["max_length"]
        )
        tok.word2idx = {k: int(v) for k, v in data["word2idx"].items()}
        tok.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        tok.vocab_size = data["vocab_size"]
        return tok

    def __repr__(self) -> str:
        return (f"SimpleTokenizer(vocab_size={self.vocab_size}, "
                f"max_length={self.max_length})")
