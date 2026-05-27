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
    MultiHeadSelfAttention — tokens attend to each other (cross-token mixing)
    TransformerBlock       — attention + FFN + residuals + layer norm
    MiniIntentEmbedder     — full model: embeddings → blocks → [CLS] → L2 norm
    build_model            — factory helper to construct the embedder
"""
import math
from dataclasses import dataclass
from typing import List, Optional

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


class TransformerBlock(nn.Module):
    """
    One transformer layer: self-attention followed by a feed-forward network (FFN).

    Attention mixes information **across tokens**. The FFN transforms each token
    **independently** with nonlinear capacity. Residual connections and layer norm
    keep training stable.

    High-level structure (Pre-Norm):
        x → norm → attention → dropout → x + residual
          → norm → FFN         → dropout → x + residual

    FFN per token:
        128 → 512 (expand) → GELU → dropout → 128 (compress)

    Residual formula (both sublayers):
        output = x + sublayer(norm(x))
    """

    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        # Cross-token mixing (see MultiHeadSelfAttention).
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        # Per-token MLP: expand → GELU → compress. No cross-token communication.
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),   # D → ffn_dim (e.g. 128 → 512)
            nn.GELU(),                        # nonlinearity (smooth ReLU-like)
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),   # ffn_dim → D (e.g. 512 → 128)
        )
        # Pre-norm: normalize before each sublayer to stabilize activations.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full data flow ([B, L, D] throughout):

            x
             ↓  norm1
             ↓  attention (tokens talk to each other)
             ↓  dropout
             ↓  x = x + ...          ← residual: preserve input, add delta
             ↓  norm2
             ↓  ff (each token 128→512→128)
             ↓  dropout
             ↓  x = x + ...          ← residual again
            out

        Args:
            x: token vectors [B, L, D]
            attention_mask: padding mask [B, L], passed through to attention

        Returns:
            Updated token vectors [B, L, D] — same shape, richer after
            both cross-token mixing and per-token transformation.
        """
        # Sublayer 1: self-attention with residual (skip connection).
        # Model learns the *change* (delta) rather than a full rewrite of x.
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        # Sublayer 2: feed-forward with residual.
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class MiniIntentEmbedder(nn.Module):
    """
    Full intent embedding model: token IDs → unit-length sentence vectors.

    Pipeline
    --------
        input_ids [B, L]
            → word_emb + pos_emb     integer IDs → vectors (+ position)
            → n_layers × TransformerBlock
            → x[:, 0, :]             pool [CLS] token at position 0
            → L2 normalize             cosine similarity = dot product

    word_emb vs Q/K/V
    -----------------
    word_emb is the **entry point** — a lookup table that maps token IDs to
    vectors. It runs *before* attention. Q/K/V projections inside attention
    operate on word_emb output, not on raw IDs.

        token ID → word_emb → [128-dim vector] → q_proj/k_proj/v_proj → Q/K/V

    Both word_emb and pos_emb are **learned** during training (start random,
    updated by backprop like any other weight).

    Shape trace ("I lost my card", B=1, L=4)
    -----------------------------------------
        input_ids:           (1, 4)
        word_emb + pos_emb:  (1, 4, 128)
        TransformerBlock ×2: (1, 4, 128) → (1, 4, 128)
        x[:, 0, :]:          (1, 128)     [CLS] only
        L2 normalize:        (1, 128)     length = 1.0
    """

    def __init__(self, config: MiniIntentConfig):
        super().__init__()
        self.config = config

        # Lookup table (vocab_size, D): token ID → D-dim vector. Not Q/K/V —
        # just converts integers into the vector space attention operates on.
        # padding_idx=0 keeps PAD row permanently zero.
        self.word_emb = nn.Embedding(
            config.vocab_size, config.embed_dim,
            padding_idx=config.pad_token_id,
        )
        # Lookup table (max_seq_len, D): position index → D-dim vector.
        # Attention has no built-in order; pos_emb encodes "I am token k".
        self.pos_emb = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.emb_norm = nn.LayerNorm(config.embed_dim)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Stack n_layers TransformerBlocks (default 2).
        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads,
                             config.ffn_dim, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.output_norm = nn.LayerNorm(config.embed_dim)
        self._init_weights()

    def _init_weights(self):
        """
        Sensible random init before training. Bad init → slow or unstable training.

        Linear:     Xavier uniform — scale by layer size, avoid exploding/vanishing
        Embedding:  Normal(std=0.02) — GPT/BERT-style small random values
        LayerNorm:  weight=1, bias=0
        PAD row:    forced to zero after init
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Flow:
            1. positions = [0, 1, ..., L-1]
            2. x = word_emb(ids) + pos_emb(positions)   per-token vectors
            3. emb_norm → emb_dropout
            4. for layer in layers: x = layer(x, mask)
            5. cls_emb = output_norm(x[:, 0, :])         [CLS] at index 0
            6. return L2-normalized cls_emb             unit vectors for retrieval

        Args:
            input_ids: token IDs from tokenizer [B, L]
            attention_mask: 1=real token, 0=pad [B, L]

        Returns:
            Sentence embeddings [B, D], L2-normalized (each row has norm 1.0)
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Position indices [0, 1, 2, ..., L-1] broadcast across batch.
        positions = torch.arange(L, device=device).unsqueeze(0)
        # Combine word meaning + position: e.g. "lost" at pos 1 knows both.
        x = self.word_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_norm(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        # Pool sentence representation from [CLS] token (always at position 0).
        cls_emb = self.output_norm(x[:, 0, :])
        # L2 norm → cosine similarity equals dot product at retrieval time.
        return F.normalize(cls_emb, p=2, dim=-1)

    def n_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())

    def encode(
        self,
        texts: List[str],
        tokenizer,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        """
        Convenience method for inference on raw text strings.

        Batches texts, tokenizes via SimpleTokenizer.encode_batch, runs forward
        under torch.no_grad() (no gradient tracking — saves memory at inference).

        Returns:
            numpy array [num_texts, embed_dim] for scikit-learn / retrieval tools
        """
        self.eval()
        all_embs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                ids, mask = tokenizer.encode_batch(batch)
                ids, mask = ids.to(device), mask.to(device)
                embs = self(ids, mask)
                all_embs.append(embs.cpu())

        return torch.cat(all_embs, dim=0).numpy()


def build_model(config: Optional[MiniIntentConfig] = None) -> MiniIntentEmbedder:
    """Build MiniIntentEmbedder with default or custom config."""
    if config is None:
        config = MiniIntentConfig()
    return MiniIntentEmbedder(config)
