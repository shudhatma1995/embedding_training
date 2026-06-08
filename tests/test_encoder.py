"""FastTextEncoder (lever C): a frozen word-embedding table + a learned [CLS], a
drop-in for MiniIntentEmbedder. Tested with a tiny SYNTHETIC matrix — no 1GB fastText
model and no models/ft_cache, so this runs in CI."""

import numpy as np
import torch
from encoder import FastTextEncoder, make_ft_tokenizer, random_matrix_like
from tokenizer import SPECIAL_TOKENS


def _matrix(v=12, d=16):
    m = np.random.default_rng(0).normal(0, 0.1, size=(v, d)).astype(np.float32)
    m[0] = 0.0  # [PAD] row zeros
    return m


def test_word_embeddings_frozen_cls_trainable():
    enc = FastTextEncoder(_matrix(), n_heads=2, n_layers=1, ffn_dim=32, max_seq_len=8)
    assert enc.word_emb.weight.requires_grad is False  # priors are frozen
    assert enc.cls.requires_grad is True  # [CLS] is learned
    # n_parameters counts ONLY trainable params → excludes the frozen word_emb table
    assert enc.n_parameters() < sum(p.numel() for p in enc.parameters())


def test_forward_shape_and_l2_norm():
    d = 16
    enc = FastTextEncoder(_matrix(d=d), n_heads=2, n_layers=1, ffn_dim=32, max_seq_len=8).eval()
    ids = torch.randint(0, 12, (3, 8))
    ids[:, 0] = SPECIAL_TOKENS["[CLS]"]
    out = enc(ids, torch.ones(3, 8, dtype=torch.long))
    assert out.shape == (3, d)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)  # L2-normalized rows


def test_make_ft_tokenizer_maps_known_and_unknown():
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "play": 3, "jazz": 4}
    tok = make_ft_tokenizer(vocab, max_seq_len=8)
    ids = tok.encode("play jazz zzznovel")
    assert ids[0] == SPECIAL_TOKENS["[CLS]"]
    assert ids[1] == 3 and ids[2] == 4  # known words → their rows
    assert ids[3] == SPECIAL_TOKENS["[UNK]"]  # unknown word → [UNK]
    assert len(ids) == 8


def test_random_matrix_like_shape_and_pad_zero():
    base = _matrix()
    r = random_matrix_like(base, seed=1)
    assert r.shape == base.shape
    assert r.dtype == np.float32
    assert np.allclose(r[0], 0.0)  # PAD row stays zeros


def test_encode_returns_numpy_rows():
    enc = FastTextEncoder(_matrix(), n_heads=2, n_layers=1, ffn_dim=32, max_seq_len=8)
    tok = make_ft_tokenizer(
        {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "play": 3, "jazz": 4}, max_seq_len=8
    )
    out = enc.encode(["play jazz", "jazz"], tok, device="cpu")
    assert out.shape == (2, enc.embed_dim)
