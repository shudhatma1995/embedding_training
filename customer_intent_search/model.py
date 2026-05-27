"""
Customer intent embedding model.

High-level pipeline
-------------------
    token IDs  →  embedding lookup  →  self-attention  →  pool [CLS]  →  sentence vector
    [B, L]         [B, L, D]            [B, L, D]         [B, D]

Tensor notation
---------------
    B  — batch size (number of sentences processed together)
    L  — sequence length (token positions; padded to max_seq_len, default 32)
    D  — embedding dimension (embed_dim, default 128)
    H  — number of attention heads (n_heads, default 4)
    Dh — head dimension (D // H, default 32)

Example: 3 queries in a batch → x.shape = [3, 32, 128] = [B, L, D]

Components in this file
-----------------------
    MiniIntentConfig       — hyperparameters shared with the tokenizer
    MultiHeadSelfAttention — core attention block (tokens attend to each other)
    (MiniIntentEmbedder)   — full model; added in a later step
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniIntentConfig:
    """
    Hyperparameters for MiniIntentEmbedder.

    Keep max_seq_len and pad_token_id in sync with SimpleTokenizer.
    Defaults target short banking-style queries (~32 tokens, ~5k vocab).
    """

    # How many words in the vocabulary in the embedding table.
    vocab_size: int = 5000
    # Every token is represented by a vector of this length (D).
    embed_dim: int = 128
    # the attention mechanism runs 4 parallel "heads", each looking at the sequence from a different angle. Each head works on 128/4 = 32 dimensions.
    n_heads: int = 4
    # the sequence passes through 2 transformer blocks stacked on top of each other. More layers = deeper understanding but slower training.
    n_layers: int = 2
    #  inside each transformer block there's a feed-forward network that expands to 512 dimensions then compresses back to 128. This is where most "thinking" happens.
    ffn_dim: int = 512
    # must match the tokenizer's max_seq_len (L).
    max_seq_len: int = 32
    # randomly zeros out 10% of values during training. Prevents overfitting.
    dropout: float = 0.1
    # tells the embedding layer to keep the padding token's vector as all zeros.
    pad_token_id: int = 0


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention: each token in the sequence attends to every token
    (including itself) and builds a new representation from weighted combinations
    of the others.

    Intuition
    ---------
    For "what is my balance", the vector for "balance" should incorporate
    context from "my" and "what". Self-attention learns those links via
    learned similarity scores between token pairs.

    Q / K / V roles
    ---------------
    Q (query)  — what a token is looking for
    K (key)    — what a token advertises for matching
    V (value)  — what information gets passed if a token is attended to

    Self-attention formula (per head):
        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Multi-head (H heads, head dimension d_k = embed_dim / H):
        head_h = Attention(Q_h, K_h, V_h)
        MultiHead(x) = Concat(head_1, ..., head_H) W_O

    Here Q, K, V are learned linear projections of the input x:
        Q = x W_Q,  K = x W_K,  V = x W_V
    W_Q, W_K, W_V are implemented as q_proj, k_proj, v_proj; W_O is out_proj.

    Shape flow (defaults: B=batch, L=32, D=128, H=4, Dh=32)
    ---------------------------------------------------------
        x              [B, L, D]
        Q, K, V        [B, H, L, Dh]   after projection + head split
        attn scores    [B, H, L, L]     token i → token j weight in head h
        output         [B, L, D]        context-aware token vectors
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        # Each head operates on this many dimensions (e.g. 128 / 4 = 32).
        self.head_dim = embed_dim // n_heads

        # Q: "what am I looking for?" — used to score against keys.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # K: "what do I advertise for matching?" — compared to queries.
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # V: "what information do I pass if attended to?" — mixed by weights.
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Merges concatenated head outputs back to embed_dim.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # sqrt(d_k) scaling prevents dot products from growing too large
        # (which would push softmax into near-one-hot gradients).
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        High-level flow:
            1. Project input x into Q, K, V
            2. Split Q, K, V into n_heads parallel subspaces
            3. Compute attention scores: Q K^T / sqrt(d_k)
            4. Mask padded positions (set score to -inf before softmax)
            5. Softmax → attention weights; dropout on weights
            6. Weighted sum of values: weights @ V
            7. Merge heads and apply output projection W_O

        Formula (per head h):
            scores = Q_h @ K_h^T / sqrt(Dh)
            weights = softmax(scores, dim=-1)        # each row sums to 1
            head_out = weights @ V_h
            out = Concat(head_1, ..., head_H) @ W_O

        Args:
            x: token embeddings [B, L, D]
               B = batch size, L = seq_len, D = embed_dim
            attention_mask: 1 for real tokens, 0 for padding [B, L]
               comes from SimpleTokenizer.encode_batch(); prevents attending to PAD

        Returns:
            Updated token representations [B, L, D] — same shape as input,
            but each position now mixes information from the whole sentence.
        """
        # B = batch, L = sequence length, D = embedding dim (see module docstring).
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        # Reshape [B, L, D] → [B, L, H, Dh] → [B, H, L, Dh] so each head
        # has its own [L, Dh] slice of Q, K, V.
        def reshape(t):
            return t.view(B, L, H, Dh).transpose(1, 2)

        # Step 1–2: linear projections, then split into heads.
        # Q, K, V each: [B, H, L, Dh]
        Q = reshape(self.q_proj(x))
        K = reshape(self.k_proj(x))
        V = reshape(self.v_proj(x))

        # Step 3: scores[b, h, i, j] = how much token i attends to token j
        # in head h.  Formula: Q K^T / sqrt(d_k)  →  shape [B, H, L, L]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Step 4: zero out padding keys so softmax assigns them weight 0.
        # mask [B, 1, 1, L] broadcasts over batch heads and query positions.
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # Step 5: row-wise softmax → each token's weights over all keys sum to 1.
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Step 6: weighted sum of value vectors per head.
        # [B, H, L, L] @ [B, H, L, Dh] → [B, H, L, Dh]
        out = torch.matmul(attn, V)

        # Step 7: merge heads [B, H, L, Dh] → [B, L, D], then W_O.
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)