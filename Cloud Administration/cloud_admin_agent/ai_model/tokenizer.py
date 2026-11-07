"""Bag-of-words/n-gram tokenizer and vectorizer (from scratch)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    toks = TOKEN_RE.findall(text.lower())
    return toks


def features(text: str, ngrams: int = 2) -> List[str]:
    toks = tokenize(text)
    feats = list(toks)
    if ngrams >= 2:
        feats.extend(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
    return feats


class BoWVectorizer:
    def __init__(self, min_freq: int = 1, max_vocab: int = 4096, ngrams: int = 2):
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.ngrams = ngrams
        self.vocab: Dict[str, int] = {}

    def fit(self, texts: Sequence[str]) -> "BoWVectorizer":
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(features(t, self.ngrams))
        items = [(tok, c) for tok, c in counts.items() if c >= self.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        self.vocab = {tok: i for i, (tok, _) in enumerate(items[: self.max_vocab])}
        return self

    @property
    def size(self) -> int:
        return len(self.vocab)

    def transform(self, text: str) -> List[float]:
        vec = [0.0] * len(self.vocab)
        feats = features(text, self.ngrams)
        if not feats:
            return vec
        for tok in feats:
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def to_dict(self) -> dict:
        return {"min_freq": self.min_freq, "max_vocab": self.max_vocab, "ngrams": self.ngrams, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "BoWVectorizer":
        obj = cls(min_freq=d.get("min_freq", 1), max_vocab=d.get("max_vocab", 4096), ngrams=d.get("ngrams", 2))
        obj.vocab = {k: int(v) for k, v in (d.get("vocab") or {}).items()}
        return obj

