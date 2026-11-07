"""Train the from-scratch cloud admin intent model.

Builds the model locally from skills-derived examples (official cloud
doc catalogs fetched online; see SOURCES.md) and saves weights to
`cloud_admin_model.json`. No pre-trained or downloaded models are used.

NOTE: Run this only when you explicitly want to (re)train. It is not run
automatically.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .dataset import CATALOG_PATH, get_examples
from .inference import CloudIntentModel, DEFAULT_MODEL_PATH
from .network import MLP
from .tokenizer import BoWVectorizer

ROOT = Path(__file__).resolve().parents[2]


def split(examples, ratio: float = 0.85, seed: int = 42):
    rnd = random.Random(seed)
    items = list(examples)
    rnd.shuffle(items)
    n = max(1, int(len(items) * ratio))
    n = min(n, len(items) - 1) if len(items) > 1 else len(items)
    return items[:n], items[n:]


def evaluate(model: CloudIntentModel, examples) -> dict:
    if not examples:
        return {"n": 0, "top1": 0.0, "top3": 0.0}
    top1 = top3 = 0
    for text, label in examples:
        preds = model.predict_topk(text, k=3)
        ranked = [l for l, _ in preds]
        top1 += int(ranked[:1] == [label])
        top3 += int(label in ranked)
    return {"n": len(examples), "top1": top1 / len(examples), "top3": top3 / len(examples)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the from-scratch cloud admin intent model (local only).")
    p.add_argument("--out", default=str(DEFAULT_MODEL_PATH), help="Output model JSON path")
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--l2", type=float, default=0.0001)
    p.add_argument("--max-per-service", type=int, default=40)
    p.add_argument("--eval-split", type=float, default=0.85)
    p.add_argument("--ngrams", type=int, default=2)
    args = p.parse_args(argv)

    all_examples = get_examples(max_per_service=args.max_per_service)
    train_ex, holdout_ex = split(all_examples, ratio=args.eval_split, seed=42)

    texts = [t for t, _ in train_ex]
    raw_labels = [y for _, y in train_ex]
    labels = sorted(set(raw_labels))
    y = [labels.index(lbl) for lbl in raw_labels]

    vec = BoWVectorizer(min_freq=1, max_vocab=8000, ngrams=args.ngrams).fit(texts)
    X = [vec.transform(t) for t in texts]

    net = MLP(input_size=vec.size, hidden_size=args.hidden, output_size=len(labels), seed=42)
    history = net.fit(X, y, epochs=args.epochs, lr=args.lr, seed=42, l2=args.l2)

    sources = []
    for sp in [ROOT / "skills_data" / "summary.json", ROOT / "SOURCES.md"]:
        if sp.name == "summary.json" and sp.exists():
            try:
                sources = json.loads(sp.read_text()).get("sources", [])
            except Exception:
                pass
    extra = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "catalog_path": str(CATALOG_PATH),
        "training": {
            "examples_total": len(all_examples),
            "train": len(train_ex),
            "holdout": len(holdout_ex),
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "ngrams": args.ngrams,
            "max_per_service": args.max_per_service,
        },
    }
    model = CloudIntentModel(vec, net, labels, metadata=extra)
    # save with metadata embedded (CloudIntentModel.save merges extra)
    out = model.save(args.out, hidden_size=args.hidden, extra=extra)

    train_m = evaluate(model, train_ex)
    hold_m = evaluate(model, holdout_ex)
    print(f"Saved from-scratch model to: {out}")
    print(f"Data: {len(all_examples)} examples ({train_m['n']} train / {hold_m['n']} holdout) from official online catalogs")
    print(f"Vocab: {vec.size} (unigrams+bigrams) | Labels: {len(labels)} | Hidden: {args.hidden} | Epochs: {args.epochs} | L2: {args.l2}")
    print(f"Final loss: {history[-1]:.4f}")
    print(f"Train  top1: {train_m['top1']*100:.2f}% | top3: {train_m['top3']*100:.2f}%")
    print(f"Holdout top1: {hold_m['top1']*100:.2f}% | top3: {hold_m['top3']*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
