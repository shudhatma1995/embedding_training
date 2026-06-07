"""Training building blocks in train.py: the intent-set validation, the InfoNCE
loss, the warmup/decay scheduler, and the CLI defaults."""

import json

import pytest
import torch
import train


def _data_file(tmp_path, intents):
    rows = [{"text": f"{it} example {n}", "intent": it} for it in intents for n in range(3)]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(rows))
    return str(path)


# ── resolve_train_intents ─────────────────────────────────────────────────────
def test_resolve_defaults_to_real_intents_present_in_data(tmp_path):
    from intents import REAL_INTENTS

    path = _data_file(tmp_path, REAL_INTENTS)
    assert train.resolve_train_intents(None, path) == list(REAL_INTENTS)


def test_resolve_accepts_a_valid_subset(tmp_path):
    path = _data_file(tmp_path, ["media", "weather", "navigation"])
    assert train.resolve_train_intents(["media", "weather"], path) == ["media", "weather"]


def test_resolve_rejects_none_label(tmp_path):
    path = _data_file(tmp_path, ["media", "weather"])
    with pytest.raises(SystemExit):
        train.resolve_train_intents(["media", "none"], path)


def test_resolve_rejects_intent_absent_from_data(tmp_path):
    path = _data_file(tmp_path, ["media", "weather"])
    with pytest.raises(SystemExit):
        train.resolve_train_intents(["media", "smart_home"], path)


def test_resolve_rejects_fewer_than_two_intents(tmp_path):
    path = _data_file(tmp_path, ["media", "weather"])
    with pytest.raises(SystemExit):
        train.resolve_train_intents(["media"], path)


# ── mnr_loss ──────────────────────────────────────────────────────────────────
def test_mnr_loss_returns_finite_scalar():
    a = torch.eye(3)
    loss = train.mnr_loss(a, a.clone(), temperature=0.05)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_mnr_loss_lower_when_pairs_are_aligned():
    a = torch.eye(4)
    aligned = train.mnr_loss(a, a.clone(), temperature=0.05)
    # roll the positives so the diagonal is no longer the correct match
    misaligned = train.mnr_loss(a, torch.roll(a.clone(), shifts=1, dims=0), temperature=0.05)
    assert aligned < misaligned


def test_mnr_loss_accepts_extra_none_negatives():
    a = torch.eye(3)
    neg = torch.eye(3)[:2]  # two shared junk negatives → 2 extra columns
    loss = train.mnr_loss(a, a.clone(), temperature=0.05, neg_emb=neg)
    assert torch.isfinite(loss)


def test_mnr_loss_accepts_per_anchor_hard_negatives():
    a = torch.eye(3)
    # hard_neg_emb is [B*k, D]; k is inferred as 2 here (6 rows / 3 anchors)
    hard = torch.eye(3).repeat(2, 1)
    loss = train.mnr_loss(a, a.clone(), temperature=0.05, hard_neg_emb=hard)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_mnr_loss_hard_negative_raises_loss():
    # each anchor's hard negative is itself → a maximally-confusing extra column,
    # so the loss must be strictly higher than without it.
    a = torch.eye(4)
    plain = train.mnr_loss(a, a.clone(), temperature=0.05)
    with_hn = train.mnr_loss(a, a.clone(), temperature=0.05, hard_neg_emb=a.clone())
    assert with_hn > plain


# ── scheduler ─────────────────────────────────────────────────────────────────
def test_scheduler_warms_up_then_decays_to_floor():
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = train.make_scheduler(opt, warmup_steps=10, total_steps=110)

    lrs = []
    for _ in range(111):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()  # no grad → no-op, but keeps optimizer-before-scheduler order
        sched.step()

    assert lrs[0] == pytest.approx(0.0)  # cold start
    assert lrs[10] == pytest.approx(1.0)  # peak at end of warmup
    assert lrs[110] == pytest.approx(0.1)  # decayed to the 0.1x floor
    assert min(lrs[10:]) >= 0.1 - 1e-9  # never drops below the floor after warmup


# ── CLI ───────────────────────────────────────────────────────────────────────
def test_build_parser_defaults():
    args = train.build_parser().parse_args([])
    assert args.epochs == 25
    assert args.pairs_per_intent == 30
    assert args.temperature == 0.05
    assert args.seed == 42
    assert args.intents is None
    assert args.none_neg_k == 0
    assert args.none_in_vocab is False


def test_build_parser_accepts_intent_subset():
    args = train.build_parser().parse_args(["--intents", "media", "weather", "--epochs", "3"])
    assert args.intents == ["media", "weather"]
    assert args.epochs == 3
