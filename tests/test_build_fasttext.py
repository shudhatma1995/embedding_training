"""build_fasttext pure logic: union_vocab text extraction + assemble (matrix layout,
[UNK]=mean-of-kept, dropped-when-no-vector). The gensim/1GB download path is not
unit-tested (network) — assemble takes an injectable get_vector so we test it offline."""

import json

import numpy as np
import pytest
from build_fasttext import assemble, union_vocab


def test_union_vocab_normalizes_dedupes_sorts(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(
        json.dumps(
            [
                {"text": "Play JAZZ", "intent": "media"},
                {"text": "play  music!", "intent": "media"},
            ]
        )
    )
    assert union_vocab([str(p)]) == ["jazz", "music", "play"]


def test_assemble_layout_unk_mean_and_drops_missing():
    vecs = {"play": [1.0, 0.0], "jazz": [0.0, 2.0]}

    def get(w):
        return vecs.get(w)  # "novel" → None → dropped to [UNK]

    vocab, matrix, stats = assemble(["jazz", "novel", "play"], get)
    assert vocab["[PAD]"] == 0 and vocab["[UNK]"] == 1 and vocab["[CLS]"] == 2
    assert "novel" not in vocab and "play" in vocab and "jazz" in vocab
    assert matrix.shape == (5, 2)  # 3 specials + 2 kept words
    assert np.allclose(matrix[0], 0.0)  # PAD row zeros
    assert np.allclose(matrix[1], [0.5, 1.0])  # [UNK] = mean of kept vectors
    assert stats == {"total": 3, "kept": 2, "miss": 1}


def test_assemble_raises_when_no_vectors():
    with pytest.raises(ValueError):
        assemble(["a", "b"], lambda w: None)
