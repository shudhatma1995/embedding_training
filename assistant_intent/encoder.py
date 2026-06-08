"""
encoder.py  -  lever C, version (B): a FROZEN fastText embedding layer feeding a
trainable transformer. Built locally (customer_intent_search/ stays frozen).
================================================================================
Why this exists
  Lever B (subword/BPE) killed [UNK] but lost: from-scratch subword pieces can't beat
  crisp whole-word vectors on 1065 rows. The missing ingredient was PRIORS, not the
  tokenizer. Here we keep word-level tokens but replace the from-scratch embedding
  TABLE with pretrained fastText vectors (built from char n-grams -> a vector for ANY
  word, incl. OOV). The transformer + a learned [CLS] train on top.

What's frozen vs trained
  word_emb  : FROZEN fastText vectors (this is the whole point — inject priors that
              1065 rows can't learn, and give OOV words real vectors). [PAD]=zeros.
  cls       : a LEARNED [CLS] vector grafted at position 0 (fastText has no [CLS];
              the pooled token must be trainable so the transformer can use it).
  pos_emb / TransformerBlocks / norms : trained as usual.

Interface parity with MiniIntentEmbedder (forward + encode + n_parameters), so the
existing run_epoch / proto_recall1 / evaluate_model all work against it unchanged.
The transformer block itself is reused from customer_intent_search (no duplication).
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cis import SimpleTokenizer  # also inserts customer_intent_search/ on sys.path
from model import TransformerBlock  # reuse the frozen upstream block (now importable)

FT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ft_cache")


def load_ft_assets(cache=FT_CACHE):
    """Load the cached (matrix [V,D], vocab {word:id}) built by the fastText job."""
    matrix = np.load(os.path.join(cache, "matrix.npy"))
    with open(os.path.join(cache, "vocab.json")) as f:
        vocab = json.load(f)
    return matrix, vocab


def make_ft_tokenizer(vocab, max_seq_len=64):
    """A word-level tokenizer over the FIXED union vocab (so every train/eval word
    indexes its fastText row; truly novel words -> [UNK]). Reuses SimpleTokenizer's
    encode/pad/[CLS] logic by injecting the vocab instead of fitting it."""
    tok = SimpleTokenizer(max_vocab_size=len(vocab), max_seq_len=max_seq_len)
    tok.word2idx = {k: int(v) for k, v in vocab.items()}
    tok.idx2word = {int(v): k for k, v in vocab.items()}
    return tok


def random_matrix_like(matrix, seed):
    """A same-shape, same-scale RANDOM frozen matrix — the control arm that isolates
    fastText's pretrained SEMANTICS from the mere architecture change (frozen 300-d
    full-coverage embeddings). [PAD] row stays zeros."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, float(matrix.std()), size=matrix.shape).astype(np.float32)
    r[0] = 0.0
    return r


class FastTextEncoder(nn.Module):
    def __init__(
        self, embedding_matrix, n_heads=4, n_layers=2, ffn_dim=512, max_seq_len=64, dropout=0.1
    ):
        super().__init__()
        v, d = embedding_matrix.shape
        self.embed_dim = d
        self.max_seq_len = max_seq_len
        w = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.word_emb = nn.Embedding.from_pretrained(w, freeze=True, padding_idx=0)
        self.cls = nn.Parameter(torch.randn(d) * 0.02)  # learned [CLS] (grafted at pos 0)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        self.emb_norm = nn.LayerNorm(d)
        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(d, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        )
        self.output_norm = nn.LayerNorm(d)
        self._init_trainable()

    def _init_trainable(self):
        """Init the TRAINABLE params only — word_emb stays frozen fastText."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        b, length = input_ids.shape
        device = input_ids.device
        we = self.word_emb(input_ids)  # [B,L,D] frozen fastText (+ [UNK]/[PAD] rows)
        cls = self.cls.to(device).expand(b, 1, -1)  # [B,1,D] learned
        x = torch.cat([cls, we[:, 1:, :]], dim=1)  # replace pos-0 ([CLS]) with learned vector
        positions = torch.arange(length, device=device).unsqueeze(0)
        x = x + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_norm(x))
        for layer in self.layers:
            x = layer(x, attention_mask)
        cls_emb = self.output_norm(x[:, 0, :])
        return F.normalize(cls_emb, p=2, dim=-1)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, texts, tokenizer, batch_size: int = 128, device: str = "cpu"):
        """No-grad numpy path (matches MiniIntentEmbedder.encode) for the Encoder API."""
        self.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                ids, mask = tokenizer.encode_batch(texts[i : i + batch_size])
                outs.append(self(ids.to(device), mask.to(device)).cpu())
        return torch.cat(outs, dim=0).numpy()
