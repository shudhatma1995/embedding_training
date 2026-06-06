"""End-to-end smoke tests: a real (tiny) training run must complete, write its
artifacts, and produce a usable encoder + evaluation. These are slower because
they touch torch and the real data files; marked `slow` so they can be deselected
with `-m "not slow"` during quick local iteration."""

import argparse
import json
import os

import data
import numpy as np
import pytest
import train
from cis import SimpleTokenizer, build_model
from evaluate import evaluate_model
from shared import make_encoder

pytestmark = pytest.mark.slow


def test_model_embeddings_are_unit_norm():
    tok = SimpleTokenizer(max_seq_len=16).fit(["play some jazz", "turn off the lights"])
    model = build_model(tok)
    emb = model.encode(["play some jazz", "lights off"], tok, device="cpu")
    assert emb.shape == (2, model.config.embed_dim)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_train_one_epoch_writes_artifacts(tmp_path):
    out = tmp_path / "model"
    args = argparse.Namespace(
        intents=["media", "weather", "navigation"],
        epochs=1,
        pairs_per_intent=4,
        lr=3e-4,
        temperature=0.05,
        warmup_steps=2,
        none_neg_k=0,
        none_in_vocab=False,
        seed=42,
        output_dir=str(out),
    )
    model, tok = train.train(args)

    for fname in ("model.pt", "tokenizer.json", "config.json"):
        assert os.path.exists(out / fname), f"missing artifact: {fname}"

    config = json.loads((out / "config.json").read_text())
    assert config["train_intents"] == ["media", "weather", "navigation"]
    assert config["epochs"] == 1
    assert len(config["history"]) == 1

    # the saved tokenizer reloads and the model still encodes
    reloaded = SimpleTokenizer.load(str(out / "tokenizer.json"))
    assert reloaded.vocab_size == tok.vocab_size
    assert model.encode(["play some music"], reloaded, device="cpu").shape[0] == 1


def test_evaluate_model_returns_wellformed_metrics():
    intents = ["media", "weather", "navigation"]
    tok = SimpleTokenizer(max_seq_len=16)
    train_groups, test_groups, none_texts = data.load_eval_data(intents)
    tok.fit([t for qs in train_groups.values() for t in qs])
    model = build_model(tok)
    encode = make_encoder(model, tok, device="cpu")

    res = evaluate_model(encode, train_groups, test_groups, none_texts)

    assert set(res["intents"]) == set(intents)
    assert 0.0 <= res["micro"]["recall@1"] <= 1.0
    assert 0.0 <= res["honest"]["overall"] <= 1.0
    assert 0.30 <= res["honest_tau"] <= 0.95  # τ comes from the sweep grid
