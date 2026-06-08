"""SubwordTokenizer (from-scratch BPE) behaviour: special-token ids, [CLS]/pad,
the defining no-[UNK] property (unseen words fall back to subword/char pieces),
deterministic fitting, and a save/load round-trip. Also checks train.build_tokenizer
dispatches 'word' vs 'bpe'. Mirrors test_tokenizer.py since SubwordTokenizer is a
drop-in for SimpleTokenizer."""

from types import SimpleNamespace

from cis import SimpleTokenizer
from subword import SPECIAL_TOKENS, SubwordTokenizer
from train import build_tokenizer

# A pangram covers every a-z letter, so any lowercase word is built from seen chars.
PANGRAM = "the quick brown fox jumps over the lazy dog"


def test_specials_occupy_ids_0_1_2():
    tok = SubwordTokenizer().fit(["set a timer", "set an alarm"])
    assert tok.token2idx["[PAD]"] == 0
    assert tok.token2idx["[UNK]"] == 1
    assert tok.token2idx["[CLS]"] == 2
    assert tok.vocab_size == len(tok.token2idx)


def test_encode_starts_with_cls_and_pads_to_seq_len():
    # generous seq_len + short input → padding is guaranteed even at char granularity
    tok = SubwordTokenizer(max_seq_len=32).fit(["turn on the lights"])
    ids = tok.encode("turn on")
    assert ids[0] == SPECIAL_TOKENS["[CLS]"]
    assert len(ids) == 32
    assert ids[-1] == SPECIAL_TOKENS["[PAD]"]


def test_encode_truncates_long_input():
    tok = SubwordTokenizer(max_seq_len=4).fit(["one two three four five six"])
    ids = tok.encode("one two three four five six seven eight nine ten")
    assert len(ids) == 4  # truncated to max_seq_len, no padding


def test_no_unk_on_unseen_words_made_of_seen_chars():
    """The whole point of BPE: a word never seen in training still encodes to known
    pieces (down to single chars), so no information-destroying [UNK]."""
    tok = SubwordTokenizer(max_seq_len=32).fit([PANGRAM])
    ids = tok.encode("xylophonization woodworking")  # unseen words, seen chars
    content = [i for i in ids if i not in (SPECIAL_TOKENS["[CLS]"], SPECIAL_TOKENS["[PAD]"])]
    assert content  # something was produced
    assert all(i != SPECIAL_TOKENS["[UNK]"] for i in content)


def test_seen_word_round_trips_to_its_text():
    """Decoding the pieces of a seen word should reconstruct the original word."""
    tok = SubwordTokenizer().fit(["set an alarm for ten minutes", "set an alarm"])
    pieces = [tok.idx2token[i] for i in tok._encode_word("alarm")]
    assert "".join(pieces).replace("</w>", "") == "alarm"


def test_fit_is_deterministic():
    texts = ["set a timer for ten minutes", "play some jazz", "what is the weather"]
    a = SubwordTokenizer().fit(texts)
    b = SubwordTokenizer().fit(texts)
    assert a.merges == b.merges
    assert a.token2idx == b.token2idx


def test_encode_batch_shapes_and_attention_mask():
    tok = SubwordTokenizer(max_seq_len=16).fit(["call mom", "text dad now"])
    ids, mask = tok.encode_batch(["call mom", "text dad now"])
    assert ids.shape == (2, 16)
    assert mask.shape == (2, 16)
    assert (mask == (ids != SPECIAL_TOKENS["[PAD]"]).long()).all()


def test_save_load_roundtrip_preserves_encoding(tmp_path):
    tok = SubwordTokenizer(max_seq_len=12).fit(["set a timer", "set an alarm", "play jazz"])
    path = tmp_path / "tok.json"
    tok.save(str(path))
    reloaded = SubwordTokenizer.load(str(path))

    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.merges == tok.merges  # tuples restored from JSON lists
    assert reloaded.ranks == tok.ranks
    # identical ids before/after reload — the property the pipeline depends on
    assert reloaded.encode("set an alarm") == tok.encode("set an alarm")


def test_build_tokenizer_dispatches_on_kind():
    texts = ["set a timer", "play some jazz", "what is the weather"]
    # default (no attr) → word-level SimpleTokenizer
    assert isinstance(build_tokenizer(SimpleNamespace(), texts), SimpleTokenizer)
    # explicit bpe → SubwordTokenizer with the requested max_seq_len
    bpe = build_tokenizer(SimpleNamespace(tokenizer="bpe", max_seq_len=64, bpe_vocab=2000), texts)
    assert isinstance(bpe, SubwordTokenizer)
    assert bpe.max_seq_len == 64
