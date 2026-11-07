"""Inference for the from-scratch cloud admin model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from .network import MLP
from .tokenizer import BoWVectorizer

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "cloud_admin_model.json"


class CloudIntentModel:
    def __init__(self, vectorizer: BoWVectorizer, net: MLP, labels: list[str], metadata: Optional[dict] = None):
        self.vectorizer = vectorizer
        self.net = net
        self.labels = labels
        self.index = {lbl: i for i, lbl in enumerate(labels)}
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "CloudIntentModel":
        p = Path(path) if path else DEFAULT_MODEL_PATH
        data = json.loads(p.read_text(encoding="utf-8"))
        vec = BoWVectorizer.from_dict(data["vectorizer"])
        labels = data["labels"]
        net = MLP.from_dict(data["network"])
        return cls(vec, net, labels, metadata={k: v for k, v in data.items() if k not in ("vectorizer", "network", "labels")})

    def predict(self, text: str) -> Tuple[str, float]:
        x = self.vectorizer.transform(text)
        idx, probs = self.net.predict(x)
        return self.labels[idx], probs[idx]

    def predict_topk(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        x = self.vectorizer.transform(text)
        ranked = self.net.predict_topk(x, k=k)
        return [(self.labels[i], p) for i, p in ranked]

    def save(self, path: Optional[str | Path] = None, hidden_size: int = 0, extra: Optional[dict] = None) -> Path:
        p = Path(path) if path else DEFAULT_MODEL_PATH
        payload = {
            "type": "cloud-admin-intent-mlp",
            "format_version": 2,
            "from_scratch": True,
            "downloaded_model": False,
            "architecture": "BoW(unigram+bigram) -> Dense(ReLU) -> Softmax",
            "hidden_size": hidden_size or len(self.net.b1),
            "labels": self.labels,
            "vectorizer": self.vectorizer.to_dict(),
            "network": self.net.to_dict(),
        }
        if extra:
            payload.update(extra)
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p


def try_load(path: Optional[str | Path] = None) -> Optional[CloudIntentModel]:
    p = Path(path) if path else DEFAULT_MODEL_PATH
    if not p.exists():
        return None
    try:
        return CloudIntentModel.load(p)
    except Exception:
        return None
