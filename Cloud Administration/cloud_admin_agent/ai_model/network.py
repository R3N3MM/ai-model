"""A tiny feedforward neural network implemented from scratch (pure Python).

Architecture: input -> Dense(hidden, ReLU) -> Dense(output, softmax)
Training: cross-entropy loss with SGD.
No external ML libraries or downloaded weights.
"""

from __future__ import annotations

import json
import math
import random
from typing import List, Sequence, Tuple


def relu(x: List[float]) -> List[float]:
    return [v if v > 0 else 0.0 for v in x]


def relu_grad(x: List[float]) -> List[float]:
    return [1.0 if v > 0 else 0.0 for v in x]


def softmax(logits: List[float]) -> List[float]:
    m = max(logits) if logits else 0.0
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def matvec(W: List[List[float]], x: Sequence[float], b: Sequence[float]) -> List[float]:
    return [dot(row, x) + bias for row, bias in zip(W, b)]


def argmax(xs: Sequence[float]) -> int:
    best_i, best_v = 0, -1e30
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


class MLP:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 42):
        rnd = random.Random(seed)
        scale1 = (2.0 / max(1, input_size)) ** 0.5
        scale2 = (2.0 / max(1, hidden_size)) ** 0.5
        self.W1 = [[rnd.uniform(-scale1, scale1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[rnd.uniform(-scale2, scale2) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b2 = [0.0] * output_size

    def forward(self, x: List[float]) -> Tuple[List[float], List[float], List[float]]:
        z1 = matvec(self.W1, x, self.b1)
        h = relu(z1)
        z2 = matvec(self.W2, h, self.b2)
        probs = softmax(z2)
        return z1, h, probs

    def predict(self, x: List[float]) -> Tuple[int, List[float]]:
        _, _, probs = self.forward(x)
        return argmax(probs), probs

    def predict_topk(self, x: List[float], k: int = 3) -> List[Tuple[int, float]]:
        _, _, probs = self.forward(x)
        ranked = sorted(enumerate(probs), key=lambda t: t[1], reverse=True)
        return [(i, p) for i, p in ranked[: max(1, k)]]

    def _clip(self, g: float, limit: float = 5.0) -> float:
        if g > limit: return limit
        if g < -limit: return -limit
        return g

    def train_step(self, x: List[float], y: int, lr: float, l2: float = 0.0) -> float:
        z1, h, probs = self.forward(x)
        # cross-entropy grad w.r.t logits (softmax)
        dz2 = list(probs)
        dz2[y] -= 1.0
        loss = -math.log(max(probs[y], 1e-12))
        # L2 penalty (for reporting/regularization signal)
        if l2 > 0:
            l2s = 0.0
            for row in self.W1: l2s += sum(w*w for w in row)
            for row in self.W2: l2s += sum(w*w for w in row)
            loss += 0.5 * l2 * l2s

        # W2, b2 grads
        for o, g0 in enumerate(dz2):
            g = self._clip(g0)
            row = self.W2[o]
            for j in range(len(h)):
                row[j] -= lr * (g * h[j] + l2 * row[j])
            self.b2[o] -= lr * g

        # backprop to hidden
        dh = [0.0] * len(h)
        for o, g0 in enumerate(dz2):
            g = self._clip(g0)
            row = self.W2[o]
            for j in range(len(h)):
                dh[j] += g * row[j]
        dz1 = [self._clip(g * rg) for g, rg in zip(dh, relu_grad(z1))]

        # W1, b1 grads
        for j, g in enumerate(dz1):
            row = self.W1[j]
            for i in range(len(x)):
                row[i] -= lr * (g * x[i] + l2 * row[i])
            self.b1[j] -= lr * g

        return loss

    def fit(self, X: List[List[float]], y: List[int], epochs: int = 300, lr: float = 0.1, seed: int = 42, l2: float = 0.0001) -> List[float]:
        rnd = random.Random(seed)
        idxs = list(range(len(X)))
        history = []
        for _ in range(epochs):
            rnd.shuffle(idxs)
            total = 0.0
            for i in idxs:
                total += self.train_step(X[i], y[i], lr, l2=l2)
            history.append(total / max(1, len(X)))
        return history

    def to_dict(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    @classmethod
    def from_dict(cls, d: dict) -> "MLP":
        obj = cls.__new__(cls)
        obj.W1 = d["W1"]
        obj.b1 = d["b1"]
        obj.W2 = d["W2"]
        obj.b2 = d["b2"]
        return obj

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "MLP":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
