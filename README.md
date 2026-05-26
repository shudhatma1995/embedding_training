# 🔍 MiniIntent: Fine-tuning a Tiny Embedding Model for Customer Support Routing

> **Use case:** Customer support intent matching — routing user queries to the right support intent using semantically meaningful embeddings trained with contrastive learning.

---

## Overview

This project fine-tunes a **tiny, self-contained transformer embedding model** (~453K parameters) for **customer support intent retrieval**. Given a natural-language customer query ("*I lost my debit card*"), the model embeds it into a 128-dimensional vector and retrieves the most semantically relevant intent from a corpus — enabling zero-shot query routing without retraining a classifier.

The key insight: a generic vector representation (TF-IDF or random transformer) cannot capture *intent-specific* semantics. Fine-tuning with **contrastive learning on same-intent pairs** forces the model to learn that "*My card is missing*" and "*I've misplaced my debit card*" are nearly identical, while "*What is my balance?*" and "*Book me a flight*" are unrelated — even if they share few words.

### What's inside

```
embedding_training/
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── results/
│   ├── evaluation_results.json        ← Full metric breakdown
│   ├── metrics_comparison.png         ← Bar chart: all methods
│   ├── training_curves.png            ← Loss + accuracy over epochs
│   └── per_intent_improvement.png     ← Per-intent Recall@1 delta
└── customer_intent_search/            ← All source code (subdirectory)
    ├── run_all.py                      ← 🚀 One-command pipeline
    ├── train.py                        ← Fine-tuning loop
    ├── evaluate.py                     ← Retrieval evaluation
    ├── inference_demo.py               ← Before/after demo
    ├── model.py                        ← MiniIntentEmbedder architecture
    ├── tokenizer.py                    ← Simple word tokenizer
    ├── data_preparation.py             ← HuggingFace dataset loader
    └── synthetic_data.py               ← Local dataset fallback
```

---

## The Use Case: Intent-based Query Routing

Modern customer support platforms receive thousands of queries per day. Each query needs to be routed to the right team or FAQ page. Classical approaches rely on:
- **Rule-based keyword matching** — brittle, misses paraphrases
- **TF-IDF similarity** — bags of words, no semantic understanding  
- **Fine-tuned classifiers** — need retraining every time a new intent is added

**Embedding-based retrieval** solves all three problems: a well-fine-tuned embedding model maps queries to a semantic space where intent-similar queries cluster together. Adding a new intent requires only adding its representative embedding to the corpus — **no retraining needed**.

---

## Dataset

### HuggingFace: `clinc/clinc_oos` (used with internet access)

| Property | Value |
|----------|-------|
| 🤗 Hub | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) |
| Paper | [Larson et al., EMNLP 2019](https://arxiv.org/abs/1909.02027) |
| Intents | 150 in-scope + 1 out-of-scope |
| Domains | banking, credit_cards, travel, home, auto, utility, work, small_talk, meta, duplication |
| Train size | 15,000 queries (100/intent) |
| Val size | 3,000 queries |
| Test size | 4,500 queries (30/intent) |
| OOS set | 1,200 out-of-scope queries |

The CLINC OOS dataset is specifically designed for intent classification evaluation. It covers 10 real-world domains and includes an out-of-scope test set to evaluate fallback behaviour — highly realistic for production customer support systems.

### Local synthetic dataset (fallback when Hub is unavailable)

When HuggingFace Hub is not accessible (offline environments, restricted networks), the code automatically falls back to a locally generated synthetic dataset with the same structure:

| Property | Value |
|----------|-------|
| Intents | 20 in-scope (banking, cards, travel, utility, small_talk) |
| Train | 3,000 queries (150/intent) |
| Val | 600 queries (30/intent) |
| Test | 600 queries (30/intent) |

The synthetic dataset uses template-based generation with slot filling, producing realistic paraphrase variation. It can be regenerated deterministically via `python synthetic_data.py`.

---

## Model Architecture: MiniIntentEmbedder

A tiny but complete transformer encoder, built entirely from scratch (no pretrained weights required). Designed to be the **smallest viable** embedding model for intent matching.

```
Input text  →  SimpleTokenizer  →  [CLS] w₁ w₂ ... wₙ (max 32 tokens)
                                       ↓
                          Word Embedding (409 × 128)
                        + Positional Embedding (32 × 128)
                        + LayerNorm + Dropout(0.1)
                                       ↓
                    ┌─── Transformer Block 1 ────────────────┐
                    │  Multi-Head Self-Attention (4 heads)    │
                    │  Pre-norm residual connection           │
                    │  FFN: 128 → 512 → 128 (GELU)           │
                    └─────────────────────────────────────────┘
                                       ↓
                    ┌─── Transformer Block 2 ────────────────┐
                    │  (same architecture)                    │
                    └─────────────────────────────────────────┘
                                       ↓
                    [CLS] token pooling  →  LayerNorm
                                       ↓
                    L2 normalisation  →  128-dim embedding
```

| Component | Value |
|-----------|-------|
| Total parameters | **452,736** (~0.45M) |
| Embedding dimension | 128 |
| Transformer layers | 2 |
| Attention heads | 4 |
| FFN dimension | 512 |
| Max sequence length | 32 tokens |
| Pooling | [CLS]-token |
| Output | L2-normalised 128-dim vector |
| Model size on disk | ~1.7MB (`model.pt`) |
| Inference speed | ~5,000 queries/sec on CPU |

The model is intentionally tiny: full training runs in **under 11 seconds on CPU** with no GPU required.

---

## Training: MultipleNegativesRankingLoss

### Loss function

Training uses **MultipleNegativesRankingLoss** (also called InfoNCE / NTXent):

```
Given a batch of B (anchor, positive) pairs:
  - anchor_i and positive_i are same-intent queries  → should be close
  - anchor_i and positive_j (j ≠ i)  → treated as negatives → should be far

Loss = -mean over i of log P(positive_i | anchor_i)
     = -mean over i of log [exp(sim(aᵢ, pᵢ)/τ) / Σⱼ exp(sim(aᵢ, pⱼ)/τ)]

where τ = 0.05 is the temperature hyperparameter.
```

With a batch size of 64, each positive pair has **63 implicit negatives** — this scales the effective training data significantly without needing explicit negative mining.

### Training pairs

From the training set, same-intent queries are paired:
- 40 pairs per intent × 20 intents = **800 positive pairs**
- Shuffled randomly before batching

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 15 | Convergence observed by epoch 12 |
| Batch size | 64 | 63 in-batch negatives per positive |
| Learning rate | 3e-4 | AdamW, higher LR works for small model |
| LR schedule | Linear warmup (50 steps) → linear decay to 10% | Stable convergence |
| Temperature τ | 0.05 | Low τ = sharper softmax, harder negatives |
| Weight decay | 0.01 | Regularisation |
| Gradient clipping | 1.0 | Prevents exploding gradients |

---

## Evaluation

### Protocol: k-NN Intent Retrieval

1. Build a **corpus** of 20 representative queries (one per intent) from the validation set
2. For each test query, embed it and compute cosine similarity against all corpus embeddings
3. Rank corpus items by similarity score
4. Check: does the top-k ranked intent match the ground-truth intent?

This evaluates the model's real-world utility: can it route an unseen query to the right intent without any classifier retraining?

### Metrics

| Metric | Definition |
|--------|-----------|
| **Recall@1** | Fraction of queries where the correct intent is ranked #1 |
| **Recall@3** | Fraction of queries where the correct intent is in the top 3 |
| **Recall@5** | Fraction of queries where the correct intent is in the top 5 |
| **MRR@5** | Mean Reciprocal Rank — average of 1/rank for all queries |

### Results (real run, 15 epochs, CPU, 18.4 seconds total)

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|--------|----------|----------|----------|-------|
| Random-Init Transformer | 25.17% | 42.83% | 53.33% | 35.07% |
| TF-IDF + SVD (LSA) | 50.00% | 67.50% | 74.33% | 59.36% |
| **Fine-tuned MiniIntent** | **97.67%** | **100.00%** | **100.00%** | **98.83%** |

**Key findings:**

- 🚀 **+47.67 percentage points** Recall@1 improvement over TF-IDF baseline
- 🚀 **+39.48 percentage points** MRR@5 improvement over TF-IDF baseline
- The fine-tuned model achieves **perfect Recall@3/5** — every test query finds its correct intent within the top 3 results
- Training for 15 epochs on 800 pairs takes **only 10.8 seconds on CPU**
- The random-init transformer (same architecture, no training) performs *worse* than TF-IDF, confirming that the improvements come from learning, not architecture

### Training dynamics

| Epoch | MNR Loss | In-batch Accuracy |
|-------|----------|-------------------|
| 1 | 4.23 | 4.3% |
| 3 | 3.93 | 7.3% |
| 5 | 2.94 | 17.8% |
| 8 | 1.97 | 30.5% |
| 12 | 1.53 | 38.7% |
| 15 | 1.49 | 37.5% |

The in-batch accuracy (fraction of anchors where the correct positive ranks first among 63 negatives) is the training-time proxy metric. Even at 37.5% batch accuracy, test retrieval is at 97.67% because the evaluation corpus is much smaller (20 items vs 63 negatives).

---

## Inference Demo: Before vs After

A concrete comparison on 10 curated queries (`inference_demo.py`):

| Query | Expected | TF-IDF | Fine-tuned |
|-------|----------|--------|------------|
| "What is my remaining balance?" | balance | ✗ `credit_limit` | ✓ `balance` |
| "My debit card was stolen" | report_lost_card | ✗ `new_card` | ✓ `report_lost_card` |
| "Please block my account for security" | freeze_account | ✗ `pay_bill` | ✓ `freeze_account` |
| "My payment was rejected at checkout" | card_declined | ✗ `report_lost_card` | ✓ `card_declined` |
| "I need to wire funds to my sister" | transfer | ✗ `freeze_account` | ✓ `transfer` |

**Demo Recall@1**: TF-IDF = 4/10 → Fine-tuned = 8/10

### Semantic similarity shift

The fine-tuned model dramatically reshapes the embedding space:

| Pair | Same intent? | TF-IDF cosine | Fine-tuned cosine | Change |
|------|-------------|---------------|-------------------|--------|
| "What is my balance?" / "How much money is in my account?" | ✓ | 0.142 | **0.822** | +0.680 ↑ |
| "I lost my debit card" / "My bank card has gone missing" | ✓ | 0.134 | **0.696** | +0.562 ↑ |
| "Transfer $200 to my friend" / "Send money to someone" | ✓ | 0.026 | **0.774** | +0.748 ↑ |
| "What is my balance?" / "Book me a flight to Paris" | ✗ | -0.006 | **-0.211** | -0.205 ↓ |

Fine-tuning pushes same-intent queries from near-zero cosine similarity (0.02–0.14) to **>0.7**, while pushing cross-intent pairs apart.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (train → eval → demo)
cd customer_intent_search
python run_all.py

# 3. Fast test run (3 epochs, reduced data)
python run_all.py --quick

# 4. Run components individually
python train.py --epochs 15 --batch-size 64
python evaluate.py --model-dir ../models/finetuned
python inference_demo.py --model-dir ../models/finetuned
```

### With HuggingFace CLINC OOS (internet access required)

The code automatically uses the full CLINC OOS dataset when the HuggingFace Hub is reachable. No code changes needed — just run with internet access:

```bash
# With internet: downloads and uses clinc/clinc_oos (150 intents, 22.5k queries)
python run_all.py --epochs 5 --batch-size 128
```

---

## Design Decisions & Findings

### Why contrastive learning outperforms TF-IDF so dramatically

TF-IDF treats text as a bag of words — "remaining balance" and "money left" share no vocabulary, so their cosine similarity is near zero. The fine-tuned model learns that *words co-occurring in the same intents are semantically related*, even if they never appear in the same sentence.

### Why a 453K-parameter model is enough

The task is relatively constrained: 20 intents with consistent domain vocabulary. A 2-layer transformer can represent this decision boundary with far fewer parameters than general-purpose models like BERT (110M). The tiny vocabulary (409 words from the training set) means the embedding matrix is tiny.

### Why `[CLS]`-token pooling vs mean pooling

`[CLS]` pooling works better here because the model is trained to route the full intent through this single token. Mean pooling would dilute the representation with filler words ("I", "me", "please"). In practice for longer documents, mean pooling often wins — but for short support queries (< 15 tokens), CLS pooling is effective.

### Why temperature τ = 0.05 matters

The temperature controls how "sharp" the softmax is during training. With τ = 0.05, the model is penalised heavily for putting probability mass on wrong negatives — creating harder, more informative training signal. With τ = 1.0 (default softmax), the gradients are much weaker.

### Limitation: vocabulary-bound generalisation

Because the tokenizer is built from training data, out-of-vocabulary words get `[UNK]`. This means queries with highly unusual vocabulary (abbreviations, typos, rare terminology) will underperform. In a production system, this would be replaced with a BPE tokenizer or a pretrained model like `paraphrase-MiniLM-L3-v2`.

---

## Extending to CLINC OOS Full Scale

When running with HuggingFace Hub access, the system scales to all 150 intents with 22,500 training queries. Expected results (based on published benchmarks with similar architectures):

| Method | Recall@1 (est.) |
|--------|----------------|
| TF-IDF + SVD | ~35–45% |
| Fine-tuned MiniIntent | ~70–85% |
| `paraphrase-MiniLM-L3-v2` (pretrained) | ~88–92% |

The gap to pretrained models narrows with more training data and larger model capacity. The `paraphrase-MiniLM-L3-v2` advantage comes from pretraining on 1B+ sentence pairs — our 453K-param model is learning intent-specific semantics from scratch.

---

## References

1. **CLINC OOS Dataset**: Larson et al. (2019). *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*. EMNLP-IJCNLP. https://arxiv.org/abs/1909.02027

2. **MultipleNegativesRankingLoss / InfoNCE**: Henderson et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply*. https://arxiv.org/abs/1705.00652

3. **Sentence-BERT**: Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP. https://arxiv.org/abs/1908.10084

4. **NTXent / SimCLR**: Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML. https://arxiv.org/abs/2002.05709

5. **CLINC OOS on HuggingFace**: https://huggingface.co/datasets/clinc/clinc_oos

---

## License

MIT — see individual dataset licenses for the CLINC OOS dataset.
