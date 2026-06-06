"""Tokenizer behaviour: deterministic vocab, [CLS]/[PAD]/[UNK] handling, and a
save/load round-trip. SimpleTokenizer is borrowed from customer_intent_search via
cis.py; we test it through that public surface since the whole pipeline relies on
it producing identical ids before and after a reload."""

from cis import SimpleTokenizer
from tokenizer import SPECIAL_TOKENS, tokenize


def test_tokenize_lowercases_and_splits_on_nonalnum():
    assert tokenize("Hello, WORLD! 123") == ["hello", "world", "123"]
    assert tokenize("don't  stop") == ["don", "t", "stop"]


def test_fit_includes_specials_and_words():
    tok = SimpleTokenizer().fit(["play some jazz", "play the news"])
    for special, idx in SPECIAL_TOKENS.items():
        assert tok.word2idx[special] == idx
    # "play" appears twice → must be in vocab.
    assert "play" in tok.word2idx
    assert tok.vocab_size == len(tok.word2idx)


def test_fit_respects_max_vocab_size():
    tok = SimpleTokenizer(max_vocab_size=5).fit(["a b c d e f g h i j"])
    assert tok.vocab_size <= 5  # 3 specials + at most 2 words


def test_encode_starts_with_cls_and_pads_to_seq_len():
    tok = SimpleTokenizer(max_seq_len=8).fit(["turn on the lights"])
    ids = tok.encode("turn on the lights")
    assert ids[0] == SPECIAL_TOKENS["[CLS]"]
    assert len(ids) == 8
    # trailing padding
    assert ids[-1] == SPECIAL_TOKENS["[PAD]"]


def test_encode_maps_unknown_to_unk():
    tok = SimpleTokenizer(max_seq_len=8).fit(["known words only"])
    ids = tok.encode("zzz totallyunseen")
    # after [CLS], every content token is unseen → [UNK]
    content = [i for i in ids[1:] if i != SPECIAL_TOKENS["[PAD]"]]
    assert content and all(i == SPECIAL_TOKENS["[UNK]"] for i in content)


def test_encode_truncates_long_input():
    tok = SimpleTokenizer(max_seq_len=4).fit(["one two three four five six"])
    ids = tok.encode("one two three four five six")
    assert len(ids) == 4  # truncated, no padding needed


def test_encode_batch_shapes_and_attention_mask():
    tok = SimpleTokenizer(max_seq_len=16).fit(["call mom", "text dad now"])
    ids, mask = tok.encode_batch(["call mom", "text dad now"])
    assert ids.shape == (2, 16)
    assert mask.shape == (2, 16)
    # mask is exactly the non-pad positions
    assert (mask == (ids != SPECIAL_TOKENS["[PAD]"]).long()).all()


def test_save_load_roundtrip_preserves_encoding(tmp_path):
    tok = SimpleTokenizer(max_seq_len=12).fit(["set a timer", "set an alarm"])
    path = tmp_path / "tok.json"
    tok.save(str(path))
    reloaded = SimpleTokenizer.load(str(path))

    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.word2idx == tok.word2idx
    # identical ids before/after reload — the property the pipeline depends on
    assert reloaded.encode("set a timer") == tok.encode("set a timer")
