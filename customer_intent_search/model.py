"""
MiniIntent Embedding Model
===========================
A small, self-contained transformer-based sentence embedding model.
Trained from scratch — no internet access or pretrained weights required.

Architecture:
  - Word embedding layer (vocab_size × embed_dim)
  - N-layer transformer encoder (BERT-style: multi-head attention + FFN)
  - [CLS]-token pooling
  - L2 normalisation output

Default configuration:
  - embed_dim:    128
  - n_heads:      4
  - n_layers:     2
  - ffn_dim:      512
  - max_seq_len:  32
  - dropout:      0.1
  Parameters: ~600k (for vocab_size=5000)

This is the "smallest viable" transformer embedding model. It is intentionally
kept tiny so the whole fine-tuning loop runs in under a minute on CPU.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniIntentConfig:
    vocab_size: int = 5000
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 512
    max_seq_len: int = 32
    dropout: float = 0.1
    pad_token_id: int = 0

    def n_parameters(self) -> int:
        """Rough parameter count."""
        embedding = self.vocab_size * self.embed_dim
        pos_embedding = self.max_seq_len * self.embed_dim
        # Per layer: attention (4 * D^2) + FFN (D * F + F * D)
        per_layer = 4 * self.embed_dim ** 2 + 2 * self.embed_dim * self.ffn_dim
        total = embedding + pos_embedding + self.n_layers * per_layer
        return total


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,                    # (B, L, D)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        def reshape(t):
            return t.view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)

        Q = reshape(self.q_proj(x))
        K = reshape(self.k_proj(x))
        V = reshape(self.v_proj(x))

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        if attention_mask is not None:
            # Convert 0→-inf so padding positions get near-zero softmax weight
            mask = attention_mask[:, None, None, :]  # (B, 1, 1, L)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)            # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer encoder block: attention → feed-forward."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm (more stable for small models)
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class MiniIntentEmbedder(nn.Module):
    """
    Small transformer sentence embedding model for customer intent matching.

    Encodes variable-length text sequences to fixed 128-dim embedding vectors.
    Uses [CLS]-token pooling for the sentence representation.

    Example usage:
        tokenizer = SimpleTokenizer(max_vocab_size=5000)
        tokenizer.fit(all_texts)

        model = MiniIntentEmbedder(MiniIntentConfig(vocab_size=tokenizer.vocab_size))

        input_ids, mask = tokenizer.encode_batch(["what's my balance?"])
        emb = model(input_ids, mask)  # (1, 128), L2-normalised
    """

    def __init__(self, config: MiniIntentConfig):
        super().__init__()
        self.config = config

        self.word_emb = nn.Embedding(
            config.vocab_size, config.embed_dim,
            padding_idx=config.pad_token_id
        )
        self.pos_emb = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.emb_norm = nn.LayerNorm(config.embed_dim)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads,
                             config.ffn_dim, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.output_norm = nn.LayerNorm(config.embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for all weight matrices."""
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
        input_ids: torch.Tensor,                       # (B, L)
        attention_mask: Optional[torch.Tensor] = None, # (B, L)
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            L2-normalised sentence embeddings of shape (B, embed_dim).
        """
        B, L = input_ids.shape
        device = input_ids.device

        positions = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
        x = self.word_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_norm(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        # [CLS]-token pooling: take position 0
        cls_emb = self.output_norm(x[:, 0, :])   # (B, D)

        # L2 normalise so cosine sim == dot product
        return F.normalize(cls_emb, p=2, dim=-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(
        self,
        texts: list,
        tokenizer,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> "torch.Tensor":
        """
        Convenience method: encode a list of texts to embeddings.

        Compatible with the retrieval evaluation pipeline.
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

        import numpy as np
        return torch.cat(all_embs, dim=0).numpy()


def build_model(tokenizer) -> MiniIntentEmbedder:
    """Build a MiniIntentEmbedder from a fitted tokenizer."""
    config = MiniIntentConfig(vocab_size=tokenizer.vocab_size)
    return MiniIntentEmbedder(config)


if __name__ == "__main__":
    config = MiniIntentConfig(vocab_size=5000)
    model = MiniIntentEmbedder(config)

    print(f"MiniIntentEmbedder")
    print(f"  Parameters:  {model.n_parameters():,}")
    print(f"  Embed dim:   {config.embed_dim}")
    print(f"  Layers:      {config.n_layers}")
    print(f"  Heads:       {config.n_heads}")

    # Test forward pass
    ids = torch.randint(0, 5000, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.long)
    out = model(ids, mask)
    print(f"  Output shape: {out.shape}")
    print(f"  Output norms: {torch.norm(out, dim=-1)}")  # Should all be ~1.0
