"""
subword.py  -  a from-scratch Byte-Pair-Encoding tokenizer (lever B: kill [UNK]).
================================================================================
Why this exists
  The word-level SimpleTokenizer maps any word it never saw in training to [UNK],
  destroying that word's information. On real user data that is common — our OOV
  diagnostic measured 11% of tokens [UNK] on in-dist templates, 17% on hand-written
  wild queries, and 27.6% on the MASSIVE real-utterance set (85% of those queries
  carry at least one [UNK]). The smoking gun: even "alarms" is [UNK], because we
  trained on "alarm" and a word-level vocab treats the two as unrelated symbols.

What BPE does instead
  It builds a vocabulary of SUBWORD pieces and falls back to single characters, so
  EVERY word — seen or not — encodes to known pieces ("alarms" -> "alarm" + "s").
  There is no [UNK] except for a never-before-seen *character*, which for our
  [a-z0-9] domain effectively never happens.

How BPE is trained (on OUR corpus only — no pretrained data, still from scratch):
  1. pre-tokenize text into words with the SAME regex + lowercasing as
     SimpleTokenizer, so the ONLY thing that differs in the A/B is the within-word
     splitting (a clean, isolated comparison).
  2. start with every word as a sequence of single characters + an end-of-word
     marker "</w>" (so a piece that ends a word differs from the same piece mid-word).
  3. repeatedly find the most frequent adjacent pair of symbols across the corpus
     and MERGE it into one new symbol; record the merge rule. Stop at the vocab budget.
  4. to encode a word later, re-apply the learned merges (highest-priority first)
     until none apply -> the subword pieces.

Drop-in replacement for SimpleTokenizer: same constructor shape, .vocab_size,
.max_seq_len, .fit / .encode / .encode_batch / .save / .load, and the SAME special
ids ([PAD]=0, [UNK]=1, [CLS]=2). So build_model(tok) and the training loop are
unchanged — only the integer IDs differ.
"""

import json
import re
from collections import Counter

import torch

SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}
END = "</w>"  # end-of-word marker symbol


def _words(text: str) -> list:
    """Same normalization as SimpleTokenizer: lowercase, runs of [a-z0-9]."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _symbols(word: str) -> tuple:
    """A word -> tuple of single-char symbols plus the end-of-word marker."""
    return tuple(word) + (END,)


def _pairs(symbols):
    """Yield adjacent symbol pairs (s0,s1),(s1,s2),... of a symbol sequence."""
    return zip(symbols[:-1], symbols[1:])


def _merge_seq(symbols: list, a: str, b: str) -> list:
    """Return symbols with every adjacent (a, b) replaced by the merged symbol a+b."""
    out, i, n = [], 0, len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class SubwordTokenizer:
    """BPE tokenizer with the SimpleTokenizer interface (see module docstring)."""

    def __init__(self, max_vocab_size: int = 2000, max_seq_len: int = 48):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.token2idx = dict(SPECIAL_TOKENS)
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.merges = []  # ordered list of merged (a, b) pairs
        self.ranks = {}  # (a, b) -> merge priority (lower = learned earlier = applied first)
        self._cache = {}  # word -> [subword ids], filled lazily during encode

    @property
    def vocab_size(self) -> int:
        return len(self.token2idx)

    def _add_token(self, tok: str) -> None:
        if tok not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[tok] = idx
            self.idx2token[idx] = tok

    def fit(self, texts) -> "SubwordTokenizer":
        """Learn the merge rules + vocabulary from a corpus of raw texts."""
        word_freq = Counter()
        for t in texts:
            word_freq.update(_words(t))
        # current symbol sequence for each unique word (mutated as we merge)
        word_syms = {w: list(_symbols(w)) for w in word_freq}
        # seed the vocab with every single-char symbol (+ END), in deterministic order
        for tok in sorted({s for syms in word_syms.values() for s in syms}):
            self._add_token(tok)
        budget = self.max_vocab_size - len(self.token2idx)  # merges we can still afford
        for _ in range(max(0, budget)):
            pair_freq = Counter()
            for w, syms in word_syms.items():
                f = word_freq[w]
                for pair in _pairs(syms):
                    pair_freq[pair] += f
            if not pair_freq:
                break
            # most frequent pair; ties broken lexicographically for reproducibility
            best = max(pair_freq, key=lambda p: (pair_freq[p], p))
            if pair_freq[best] < 2:
                break  # nothing repeats — further merges would just memorize singletons
            self.ranks[best] = len(self.merges)
            self.merges.append(best)
            self._add_token(best[0] + best[1])
            a, b = best
            for w in word_syms:
                word_syms[w] = _merge_seq(word_syms[w], a, b)
        self._cache.clear()
        return self

    def _encode_word(self, word: str) -> list:
        """Apply learned merges to one word -> list of subword ids (cached)."""
        if word in self._cache:
            return self._cache[word]
        syms = list(_symbols(word))
        while len(syms) > 1:
            # pick the adjacent pair with the best (lowest) merge rank present
            best_rank, best_i = None, None
            for i, pair in enumerate(_pairs(syms)):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break  # no mergeable pair left
            syms = syms[:best_i] + [syms[best_i] + syms[best_i + 1]] + syms[best_i + 2 :]
        ids = [self.token2idx.get(s, SPECIAL_TOKENS["[UNK]"]) for s in syms]
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list:
        """Encode text -> [CLS] + subword ids, truncated/padded to max_seq_len."""
        ids = [SPECIAL_TOKENS["[CLS]"]]
        for w in _words(text):
            ids.extend(self._encode_word(w))
        ids = ids[: self.max_seq_len]
        return ids + [SPECIAL_TOKENS["[PAD]"]] * (self.max_seq_len - len(ids))

    def encode_batch(self, texts):
        """Encode a batch -> (input_ids, attention_mask) torch LongTensors [B, L]."""
        encoded = [self.encode(t) for t in texts]
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = (input_ids != SPECIAL_TOKENS["[PAD]"]).long()
        return input_ids, attention_mask

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "kind": "bpe",
                    "max_vocab_size": self.max_vocab_size,
                    "max_seq_len": self.max_seq_len,
                    "token2idx": self.token2idx,
                    "merges": self.merges,  # JSON turns each (a,b) into [a,b]
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "SubwordTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(data["max_vocab_size"], data["max_seq_len"])
        tok.token2idx = {k: int(v) for k, v in data["token2idx"].items()}
        tok.idx2token = {v: k for k, v in tok.token2idx.items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.ranks = {pair: i for i, pair in enumerate(tok.merges)}
        return tok
