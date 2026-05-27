"""
Step 1 — What is an Embedding?
================================
Run this file, read the output, then tweak the code and re-run.
Every section is labeled so you know what concept it covers.
"""

import numpy as np
from tokenizer import SimpleTokenizer
from synthetic_data import generate_synthetic_dataset


# ── Load data ─────────────────────────────────────────────────────────────────
train, val, test = generate_synthetic_dataset(n_train=5)
all_texts = [q for qs in train.values() for q in qs]


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT 1: Vocabulary
# The tokenizer scans all training sentences and builds a lookup table:
#   word (string) → integer ID
# Only words seen during training get a real ID.
# Everything else → [UNK] (ID = 1).
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("CONCEPT 1: Vocabulary")
print("=" * 60)

tok = SimpleTokenizer(max_vocab_size=500, max_length=16)
tok.fit(all_texts)

print(f"Training sentences     : {len(all_texts)}")
print(f"Vocabulary size        : {tok.vocab_size} tokens")
print(f"Special tokens         : PAD=0  UNK=1  CLS=2")
print(f"\nFirst 15 words in vocab:")
for word, idx in list(tok.word2idx.items())[3:18]:   # skip special tokens
    print(f"  ID {idx:>4}  →  '{word}'")

# ── Try changing max_vocab_size to 50 and see what changes ───────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT 2: Tokenization
# Text → list of integer IDs.
# The model never sees raw characters — only these IDs.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONCEPT 2: Tokenization")
print("=" * 60)

sentences = [
    "What is my balance?",
    "How much money do I have?",
    "WHAT IS MY BALANCE?",        # uppercase — same result?
    "my balance what is",         # same words, shuffled
    "xyzabc unknown words!!!",    # totally unknown
]

for s in sentences:
    ids   = tok.encode(s)
    words = [tok.idx2word[i] for i in ids]
    print(f"\n  Input : {s}")
    print(f"  IDs   : {ids}")
    print(f"  Words : {words}")

# ── Questions to think about ──────────────────────────────────────────────────
# 1. What happens to uppercase letters?
# 2. "my balance what is" — same IDs as the original sentence?
#    If yes, is that a problem?
# 3. "xyzabc" maps to [UNK]. What does the model "see" for unknown words?


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT 3: Bag-of-Words vector
# The simplest possible "embedding": count how many times each vocab word
# appears in the sentence → one big sparse vector.
# This is the foundation of TF-IDF.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONCEPT 3: Bag-of-Words vector")
print("=" * 60)

def bow_vector(text: str) -> np.ndarray:
    """Count-based vector — one dimension per vocabulary word."""
    v = np.zeros(tok.vocab_size)
    for idx in tok.encode(text):
        v[idx] += 1.0
    return v

sentence = "What is my balance?"
vec = bow_vector(sentence)
nonzero = [(tok.idx2word[i], vec[i]) for i in np.where(vec > 0)[0]]

print(f"\nSentence : '{sentence}'")
print(f"Vector length : {len(vec)}  (one slot per vocab word)")
print(f"Non-zero entries : {nonzero}")
print(f"All other {len(vec) - len(nonzero)} dimensions are 0.0")


# ─────────────────────────────────────────────────────────────────────────────
# CONCEPT 4: Cosine Similarity
# Given two vectors A and B:
#   cosine_sim(A, B) = (A · B) / (|A| × |B|)
# Range: -1 (opposite) → 0 (unrelated) → +1 (identical direction)
#
# If vectors are L2-normalised (length = 1) first:
#   cosine_sim(A, B) = A · B   ← just a dot product
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONCEPT 4: Cosine Similarity on raw Bag-of-Words")
print("=" * 60)

def cosine_sim(text_a: str, text_b: str) -> float:
    """Cosine similarity between two sentences using BoW vectors."""
    a = bow_vector(text_a)
    b = bow_vector(text_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

pairs = [
    # Same intent, same words
    ("What is my balance?",
     "Check my balance please",
     "same intent, overlapping words"),
    # Same intent, DIFFERENT words
    ("What is my balance?",
     "How much money do I have?",
     "same intent, NO shared words"),
    # Different intent, coincidental word overlap
    ("What is my balance?",
     "What is the exchange rate?",
     "different intent, shares 'what is'"),
    # Clearly different
    ("What is my balance?",
     "Book me a flight to Paris",
     "clearly different"),
    # Paraphrases — the hard case
    ("I lost my debit card",
     "My bank card has gone missing",
     "same intent, zero word overlap"),
]

print(f"\n  {'Score':>7}  Description")
print("  " + "-" * 55)
for a, b, desc in pairs:
    sim = cosine_sim(a, b)
    print(f"  {sim:>+7.3f}  {desc}")
    print(f"           A: '{a}'")
    print(f"           B: '{b}'")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# THE KEY PROBLEM — shown concretely
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("THE PROBLEM bag-of-words cannot solve")
print("=" * 60)
print("""
These two sentences mean EXACTLY the same thing:
  "I lost my debit card"
  "My bank card has gone missing"

But they share ZERO vocabulary words.
BoW cosine similarity = 0.000

A good embedding model must learn that 'lost' ≈ 'missing'
and 'debit card' ≈ 'bank card' — from context, not spelling.
That is what fine-tuning teaches it.
""")
