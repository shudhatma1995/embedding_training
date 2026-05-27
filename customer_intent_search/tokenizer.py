import json
import re
from collections import Counter
from typing import List, Tuple

import torch

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
}

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

class SimpleTokenizer:
    def __init__(self, max_vocab_size: int = 8000, max_seq_len: int = 32):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx = dict(SPECIAL_TOKENS)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def fit(self, texts: List[str]) -> "SimpleTokenizer":
        """Fit the tokenizer on a list of texts."""
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        
        n_special = len(SPECIAL_TOKENS)
        n_top = self.max_vocab_size - n_special
        most_common = counter.most_common(n_top)
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self


    def encode(self, text: str) -> List[int]:
        """Encode a text into a list of token IDs."""
        tokens = tokenize(text)
        ids = [SPECIAL_TOKENS["[CLS]"]]
        for token in tokens:
            if token not in self.word2idx:
                token = "[UNK]"
            ids.append(self.word2idx[token])
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
        return ids + [SPECIAL_TOKENS["[PAD]"]] * (self.max_seq_len - len(ids))


    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts into a list of token IDs."""
        encoded = [self.encode(text) for text in texts]

        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = (input_ids != SPECIAL_TOKENS["[PAD]"]).long()
        return input_ids, attention_mask

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "max_vocab_size": self.max_vocab_size,
                "max_seq_len": self.max_seq_len,
                "word2idx": self.word2idx,
            }, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(data["max_vocab_size"], data["max_seq_len"])
        tok.word2idx = {k: int(v) for k, v in data["word2idx"].items()}
        tok.idx2word = {v: k for k, v in tok.word2idx.items()}
        return tok

# test
if __name__ == "__main__":

    print("\n=== test fit ===")
    texts = ["check my balance", "my balance is low", "transfer money"]
    tok = SimpleTokenizer()
    tok.fit(texts)
    print(tok.word2idx)
    # {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "my": 3, "balance": 4, "check": 5, ...}
    print(tok.vocab_size)  # 11


    # test encode
    print("\n=== test encode ===")
    tok = SimpleTokenizer()
    tok.fit(["what is my balance", "transfer money to john"])
    print("what is my balance: ", tok.encode("what is my balance"))
    print("unknown xyzword: ", tok.encode("unknown xyzword"))
    print("transfer money to john: ", tok.encode("transfer money to john"))

    # test encode_batch
    print("\n=== test encode_batch ===")
    texts = ["what is my balance", "lost my card", "transfer money to john please"]
    tok = SimpleTokenizer()
    tok.fit(texts)
    ids, mask = tok.encode_batch(texts)
    print(ids.shape)   # torch.Size([3, 32])
    print(mask.shape)  # torch.Size([3, 32])
    print(ids[0])      # first sentence as token IDs
    print(mask[0])     # 1s followed by 0s

    # test save and load
    print("\n=== test save and load ===")
    from pathlib import Path
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    tokenizer_path = data_dir / "tokenizer.json"
    tok = SimpleTokenizer()
    tok.fit(["what is my balance", "lost my card", "transfer money"])
    tok.save(tokenizer_path)
    tok2 = SimpleTokenizer.load(tokenizer_path)
    print(tok2.vocab_size)
    print(tok2.encode("my balance"))
    print(tok2.encode("unknown xyzword"))
    print(tok2.encode("transfer money to john"))
