# embedding_training

Training and evaluation code for embedding models, starting with customer intent search.

## Project layout

```
embedding_training/
├── customer_intent_search/   # Intent search package
│   ├── data/                 # Saved tokenizer and datasets
│   └── tokenizer.py          # SimpleTokenizer implementation
├── models/                   # Trained model checkpoints
├── results/                  # Evaluation plots and metrics
└── requirements.txt
```

## Setup

```bash
cd embedding_training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Tokenizer concepts

A **tokenizer** converts raw text into numbers that a neural network can process. Models do not read strings directly—they operate on integer **token IDs** and fixed-size tensors.

This project uses `SimpleTokenizer`, a lightweight **word-level** tokenizer built for short customer-support queries (for example, "what is my balance" or "lost my card").

### Text → tokens → IDs

The pipeline has three steps:

1. **Normalize and split** — lowercase the text and split on word boundaries (`tokenize()`).
2. **Map words to IDs** — look up each word in the vocabulary (`word2idx`).
3. **Pad to fixed length** — every sequence is padded or truncated to `max_seq_len` (default 32).

Example:

```
"my balance"  →  ["my", "balance"]  →  [CLS] my balance [PAD] [PAD] ...
```

### Vocabulary (`fit`)

Before encoding, the tokenizer must **learn** which words exist in your training data.

- Call `fit(texts)` on a list of example sentences.
- The tokenizer counts word frequencies and keeps the most common words up to `max_vocab_size` (default 8000).
- Rare words are not stored individually; at encode time they map to the unknown token.

The vocabulary is a dictionary: `word2idx` maps strings to integers, and `idx2word` is the reverse lookup.

### Special tokens

Three reserved tokens are always present at fixed IDs:

| Token | ID | Purpose |
|-------|----|---------|
| `[PAD]` | 0 | Fills unused positions so every sequence has the same length |
| `[UNK]` | 1 | Substitute for words not in the vocabulary |
| `[CLS]` | 2 | Prepended to every sequence as a sentence-level marker |

**Padding** lets batches stack into rectangular tensors. Positions filled with `PAD` should be ignored by the model.

**Unknown (`UNK`)** handles out-of-vocabulary words at inference time (typos, names, new phrases).

**CLS** (classification token) gives the model a consistent starting point for reading a sentence—useful when pooling token representations into one embedding.

### Encoding a single sentence (`encode`)

`encode(text)` returns a list of exactly `max_seq_len` integers:

1. Prepend `[CLS]`.
2. Convert each word to its ID, or `UNK` if missing.
3. Truncate if longer than `max_seq_len`.
4. Pad the rest with `PAD`.

```python
from customer_intent_search import SimpleTokenizer

tok = SimpleTokenizer()
tok.fit(["what is my balance", "lost my card", "transfer money"])

tok.encode("my balance")
# [2, 3, 4, 0, 0, ...]  →  [CLS] my balance PAD PAD ...

tok.encode("unknown xyzword")
# [2, 1, 1, 0, 0, ...]  →  [CLS] UNK UNK PAD PAD ...
```

### Batching (`encode_batch`)

Neural networks train on **batches** of examples. `encode_batch(texts)` encodes multiple sentences and returns two PyTorch tensors:

- **`input_ids`** — shape `[batch_size, max_seq_len]`, the token IDs
- **`attention_mask`** — shape `[batch_size, max_seq_len]`, `1` for real tokens and `0` for padding

The attention mask tells the model which positions are meaningful and which are padding.

```python
texts = ["what is my balance", "lost my card"]
input_ids, attention_mask = tok.encode_batch(texts)
# input_ids.shape      → torch.Size([2, 32])
# attention_mask.shape → torch.Size([2, 32])
```

### Save and load

After fitting on training data, persist the vocabulary so inference uses the same word mappings:

```python
from pathlib import Path

data_dir = Path("customer_intent_search/data")
data_dir.mkdir(exist_ok=True)

tok.save(data_dir / "tokenizer.json")
tok = SimpleTokenizer.load(data_dir / "tokenizer.json")
```

The saved JSON stores `max_vocab_size`, `max_seq_len`, and `word2idx`.

### Run tests

```bash
python customer_intent_search/tokenizer.py
```

This runs built-in checks for `fit`, `encode`, `encode_batch`, and save/load.

## Dependencies

| Package | Role |
|---------|------|
| `torch` | Tensors and model training |
| `numpy` | Numerical utilities |
| `scikit-learn` | Metrics and evaluation helpers |
| `matplotlib` | Result plots |
| `tqdm` | Progress bars |
| `sentence-transformers` | Pretrained embedding models |
| `datasets` | Dataset loading |
