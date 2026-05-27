# embedding_training

Training and evaluation code for embedding models, starting with customer intent search.

This repo is built **step by step**. You can clone it, follow each step in order, run the code locally, and build the same customer-intent embedding pipeline from scratch.

## Project layout

```
embedding_training/
├── customer_intent_search/   # Intent search package
│   ├── data/                 # Saved tokenizer and datasets
│   ├── tokenizer.py          # SimpleTokenizer implementation
│   └── model.py              # Embedding model (config + attention)
├── models/                   # Trained model checkpoints
├── results/                  # Evaluation plots and metrics
└── requirements.txt
```

## End-to-end pipeline

At a high level, customer intent search turns a short query into a dense vector you can compare against known intents:

```
Raw text          Token IDs           Token vectors         Context-aware          Sentence
"my balance"  →   [2, 3, 4, 0, …]  →  [B, L, D] tensor  →  self-attention    →  embedding
                  (Step 1)              (embedding lookup)    (Step 2)             (pool [CLS])
```

| Stage | Input | Output | Implemented in |
|-------|-------|--------|----------------|
| Tokenize | string | `[L]` integer IDs + mask | `tokenizer.py` |
| Embed | token IDs | `[B, L, D]` float tensor | `model.py` |
| Attend | embeddings + mask | `[B, L, D]` updated vectors | `MultiHeadSelfAttention` |
| Pool | token vectors | `[B, D]` sentence embedding | `model.py` |

**Tensor shorthand used throughout:**

| Symbol | Name | Meaning | Typical value here |
|--------|------|---------|-------------------|
| **B** | Batch size | Number of sentences processed together | e.g. 32 |
| **L** | Sequence length | Token positions per sentence (padded) | 32 |
| **D** | Embedding dim | Numbers per token vector | 128 |
| **H** | Heads | Parallel attention subspaces | 4 |

Example: a batch of 3 queries → `x.shape = [3, 32, 128]` → `[B, L, D]`.

## Setup

```bash
cd embedding_training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Follow along

Work through the steps below in order. Each step introduces one component, with files you can read, commands to run, and concepts to understand before moving on.

| Step | Topic | Status |
|------|-------|--------|
| 1 | Tokenizer — text to token IDs | Done |
| 2 | Model — `MultiHeadSelfAttention` | Done |

---

### Step 1 — Tokenizer

**Goal:** Turn raw customer queries (for example, `"what is my balance"`) into fixed-length lists of integers that a neural network can consume.

**Key files:**
- `customer_intent_search/tokenizer.py` — `SimpleTokenizer` implementation
- `customer_intent_search/data/tokenizer.json` — saved vocabulary after fitting

#### What to do

1. **Clone the repo and install dependencies** (see [Setup](#setup) above).
2. **Open `customer_intent_search/tokenizer.py`** and skim the `SimpleTokenizer` class: `fit`, `encode`, `encode_batch`, `save`, and `load`.
3. **Run the built-in tests** from the project root:
   ```bash
   python customer_intent_search/tokenizer.py
   ```
   You should see output for `fit`, `encode`, `encode_batch`, and save/load without errors.
4. **Fit the tokenizer on your own sample texts** (short banking-style queries work well):
   ```python
   from pathlib import Path
   from customer_intent_search import SimpleTokenizer

   texts = [
       "what is my balance",
       "lost my card",
       "transfer money to john",
       "how do i reset my password",
   ]

   tok = SimpleTokenizer()
   tok.fit(texts)
   print(tok.vocab_size)
   print(tok.word2idx)
   ```
5. **Encode a few sentences** and inspect the token IDs:
   ```python
   print(tok.encode("my balance"))
   print(tok.encode("unknown xyzword"))  # out-of-vocabulary → UNK
   ```
6. **Save and reload** so the same vocabulary is reused later (training, inference):
   ```python
   data_dir = Path("customer_intent_search/data")
   data_dir.mkdir(exist_ok=True)
   tok.save(data_dir / "tokenizer.json")

   tok2 = SimpleTokenizer.load(data_dir / "tokenizer.json")
   print(tok2.encode("transfer money"))
   ```
7. **Try batch encoding** — this is what the model will use during training:
   ```python
   input_ids, attention_mask = tok.encode_batch(texts)
   print(input_ids.shape)       # [batch_size, 32]
   print(attention_mask.shape)  # [batch_size, 32]
   ```

When Step 1 is working, you should have a saved `tokenizer.json` under `customer_intent_search/data/` and understand how text becomes padded integer tensors.

#### Tokenizer concepts

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

---

### Step 2 — Model (embeddings + self-attention)

**Goal:** Map token IDs to dense vectors and let each token gather context from the rest of the sentence—so `"balance"` can incorporate `"my"` and `"check"`.

**Key files:**
- `customer_intent_search/model.py` — `MiniIntentConfig`, `MultiHeadSelfAttention`

#### What to do

1. **Complete Step 1** so you have `input_ids` and `attention_mask` tensors from the tokenizer.
2. **Open `customer_intent_search/model.py`** and read `MiniIntentConfig` — these defaults tie the model to the tokenizer:
   ```python
   from customer_intent_search.model import MiniIntentConfig

   cfg = MiniIntentConfig()
   # vocab_size=5000, embed_dim=128, n_heads=4, max_seq_len=32, pad_token_id=0
   ```
3. **Read `MultiHeadSelfAttention`** — the core mechanism that mixes information across tokens. The class docstring contains the formula and a 7-step `forward()` flow.
4. **Trace one forward pass mentally:**
   - Input `x`: `[B, L, D]` — one 128-dim vector per token per sentence
   - Project to Q, K, V; split into 4 heads of 32 dims each
   - Score every token against every token → softmax → weighted sum of values
   - Output: `[B, L, D]` — same shape, richer representations

#### Model config (`MiniIntentConfig`)

| Field | Default | Role |
|-------|---------|------|
| `vocab_size` | 5000 | Rows in the embedding lookup table |
| `embed_dim` (D) | 128 | Size of each token vector |
| `n_heads` (H) | 4 | Parallel attention patterns (128 ÷ 4 = 32 dims per head) |
| `n_layers` | 2 | Stacked transformer blocks |
| `ffn_dim` | 512 | Hidden size inside feed-forward sublayer |
| `max_seq_len` (L) | 32 | Must match tokenizer |
| `dropout` | 0.1 | Regularization during training |
| `pad_token_id` | 0 | Padding token ID from tokenizer |

#### Self-attention concepts

Self-attention asks: **for each token, which other tokens matter, and what should I take from them?**

Per attention head:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

| Projection | Role | Intuition |
|------------|------|-----------|
| **Q** (query) | What am I looking for? | `"balance"` searches for related words |
| **K** (key) | What do I offer for matching? | `"my"` advertises possessive context |
| **V** (value) | What info do I pass if selected? | Actual meaning carried forward |

**Multi-head:** run H independent attentions in parallel, then concatenate. Each head can focus on different patterns (keywords, word pairs, etc.).

**Attention mask:** positions where `attention_mask == 0` (padding) get score `-∞` before softmax, so the model never attends to `PAD` tokens.

High-level `forward()` flow:

1. Linear projections → Q, K, V
2. Split into H heads
3. Scores = QKᵀ / √d_k → shape `[B, H, L, L]`
4. Mask padding keys
5. Softmax + dropout on weights
6. Output = weights × V
7. Merge heads → output projection

#### Shape cheat sheet

```
input_ids        [B, L]           integers from tokenizer
attention_mask   [B, L]           1 = real token, 0 = pad
x (embeddings)   [B, L, D]        float vectors after lookup
Q, K, V          [B, H, L, Dh]    H=4, Dh=32
attn scores      [B, H, L, L]     token i → token j weights
output           [B, L, D]        context-aware token vectors
```

#### Full input → output flow with a real example

Let's trace the sentence **"I lost my card"** through `MultiHeadSelfAttention`.

##### Step 0 — Starting point

We have 4 words. After tokenization and embedding, each word becomes a list of 128 numbers. Think of it as a table:

```
         [128 numbers each]
"I"    → [0.2, 0.5, -0.3, ...]
"lost" → [0.8, -0.1, 0.6, ...]
"my"   → [0.1, 0.3,  0.1, ...]
"card" → [0.6, 0.2, -0.5, ...]
```

Shape of `x`: `(1 sentence, 4 words, 128 numbers)` = `(1, 4, 128)` → `[B, L, D]`

##### Step 1 — Create Q, K, V

Pass `x` through the three projections. Each word gets three versions of itself:

```
         Q (asking)        K (name tag)      V (resume)
"I"    → [0.1, 0.4, ...]  [0.3, 0.2, ...]  [0.5, 0.1, ...]
"lost" → [0.9, 0.2, ...]  [0.8, 0.7, ...]  [0.6, 0.9, ...]
"my"   → [0.2, 0.1, ...]  [0.1, 0.3, ...]  [0.2, 0.4, ...]
"card" → [0.7, 0.5, ...]  [0.6, 0.4, ...]  [0.8, 0.3, ...]
```

All still shape `(1, 4, 128)` — same size, different meaning.

In code: `Q = q_proj(x)`, `K = k_proj(x)`, `V = v_proj(x)`.

##### Step 2 — Split into 4 heads

We have 128 dimensions and 4 heads → each head gets 32 dimensions.

```
Head 1 looks at dimensions  1–32
Head 2 looks at dimensions 33–64
Head 3 looks at dimensions 65–96
Head 4 looks at dimensions 97–128
```

Each head runs the full attention process independently on its 32 dimensions. Think of 4 people reading the same sentence but each focusing on different aspects (grammar, topic, sentiment, etc.).

**Let's follow just Head 1 from here.**

##### Step 3 — Compute attention scores `Q × Kᵀ`

Head 1 asks: *how much should each word attend to every other word?*

We dot-product each word's Q against every word's K:

```
              "I"   "lost"  "my"  "card"
"I"    asks → 0.2    0.3    0.1    0.2
"lost" asks → 0.1    0.5    0.1    0.8   ← "lost" pays most attention to "card"
"my"   asks → 0.2    0.1    0.4    0.1
"card" asks → 0.1    0.7    0.1    0.3   ← "card" pays most attention to "lost"
```

This is a `(4, 4)` grid — every word against every word.

Then divide by `scale = sqrt(32) ≈ 5.6` to keep numbers small.

In code: `attn = Q @ K.transpose(-2, -1) / scale` → shape `[B, H, L, L]`.

##### Step 4 — Apply attention mask

Our sentence has no padding here, so all 4 positions are real. Mask is all 1s — nothing to block.

If we had padding like `"I lost my card [PAD]"`:

```
"card" asks → 0.1   0.7   0.1   0.3   -inf  ← PAD gets -inf
```

In code: `attn.masked_fill(attention_mask == 0, -inf)`.

##### Step 5 — Softmax → weights that sum to 1

```
"lost" row before softmax: [0.1,  0.5,  0.1,  0.8]
"lost" row after softmax:  [0.12, 0.23, 0.12, 0.53]
                                                ↑
                                      "card" gets 53% of attention
```

```
"card" row before softmax: [0.1,  0.7,  0.1,  0.3]
"card" row after softmax:  [0.13, 0.48, 0.13, 0.26]
                                    ↑
                          "lost" gets 48% of attention
```

"lost" and "card" are now paying high attention to each other. The model is learning they belong together.

##### Step 6 — Weighted sum of values `weights × V`

Now each word collects information from all other words, weighted by attention:

```
new "card" =
    0.13 × V("I")    +
    0.48 × V("lost") +   ← heavy contribution from "lost"
    0.13 × V("my")   +
    0.26 × V("card")

= [0.13×0.5, 0.13×0.1, ...]
+ [0.48×0.6, 0.48×0.9, ...]   ← dominates
+ [0.13×0.2, 0.13×0.4, ...]
+ [0.26×0.8, 0.26×0.3, ...]

= [0.58, 0.55, ...]   ← new "card" vector, enriched with context from "lost"
```

Before attention, "card" only knew about itself. Now it knows it's a **lost** card.

In code: `out = attn @ V`.

##### Step 7 — Reassemble all 4 heads

All 4 heads did this independently. Now concatenate their outputs:

```
Head 1 output for "card": [0.58, 0.55, ...]  32 numbers
Head 2 output for "card": [0.31, 0.72, ...]  32 numbers
Head 3 output for "card": [0.44, 0.21, ...]  32 numbers
Head 4 output for "card": [0.67, 0.18, ...]  32 numbers

Concatenated: [...all 4 heads...]  = 128 numbers again
```

Then one final `out_proj` linear layer mixes all heads together → still `128` numbers.

##### Final output

```
Input  "card": [0.6, 0.2, -0.5, ...]   ← knew nothing about context
Output "card": [0.58, 0.55, 0.41, ...]  ← now understands it's a lost card
```

Same shape `(1, 4, 128)` in, same shape `(1, 4, 128)` out — but every word is now **context-aware**.

This is everything `MultiHeadSelfAttention` does.

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
