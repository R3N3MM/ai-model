"""Training data for the cloud admin intent model.

Primary source: skills_data/catalog.json, built from official cloud
documentation catalogs fetched online (see SOURCES.md):
- https://learn.microsoft.com/en-us/azure/?product=popular
- https://azure.microsoft.com/en-us/products/
- https://docs.aws.amazon.com/
- https://aws.amazon.com/products/
- https://cloud.google.com/products

No external data or pre-trained models are downloaded.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "skills_data" / "catalog.json"

NAV = [
    "open", "go to", "navigate to", "show me", "show", "take me to", "visit",
    "launch", "access", "manage", "view", "find", "configure", "check", "admin",
    "i want to open", "i need to access", "please open", "can you open", "bring up",
    "where is", "open the console for",
]
CONTEXT = ["", " in the portal", " in console", " dashboard", " settings", " admin page", " service"]
BROWSERS = [
    "", " in firefox", " in chrome", " in google chrome", " using edge", " in brave", " using opera", " in microsoft edge",
]
CLOUD_HINTS = {
    "azure": ["azure", "microsoft azure", "azure portal", "on azure"],
    "aws": ["aws", "amazon web services", "aws console", "on aws"],
    "gcp": ["gcp", "google cloud", "google cloud platform", "gcp console", "on gcp"],
}


def _load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        return []
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("-", " ").replace("_", " ").split())


def get_examples(seed: int = 42, max_per_service: int = 64) -> List[Tuple[str, str]]:
    catalog = _load_catalog()
    if not catalog:
        return [("open azure", "azure:home"), ("open aws", "aws:home"), ("open google cloud", "gcp:home")]

    rnd = random.Random(seed)
    examples: List[Tuple[str, str]] = []
    for item in catalog:
        cloud = item["cloud"]
        label = f"{cloud}:{item['key']}"
        phrases = set()
        for val in [item.get("name", ""), item.get("key", "").replace("-", " "), item.get("category", "")]:
            if val: phrases.add(_norm(val))
        for a in item.get("aliases", []) or []:
            phrases.add(_norm(a))
        for s in item.get("skills", []) or []:
            phrases.add(_norm(s))
        phrases = [p for p in phrases if p and len(p) >= 2]
        if not phrases:
            continue

        combos = []
        hints = CLOUD_HINTS.get(cloud, [cloud])
        for ph in phrases:
            for nav in NAV[:12]:
                combos.append(f"{nav} {ph}")
                combos.append(f"{nav} {rnd.choice(hints)} {ph}" if cloud not in ph else f"{nav} {ph}")
            combos.append(ph)
            combos.append(f"{ph} {rnd.choice(hints)}")
            combos.append(f"{rnd.choice(hints)} {item.get('category','')} {ph}")
            for sk in (item.get("skills") or [])[:3]:
                combos.append(_norm(sk))
                combos.append(f"{_norm(sk)} {rnd.choice(hints)}")
        rnd.shuffle(combos)
        seen = set()
        for c in combos:
            c = " ".join(c.split())
            if not c or c in seen:
                continue
            seen.add(c)
            text = f"{c}{rnd.choice(CONTEXT)}{rnd.choice(BROWSERS)}".strip()
            examples.append((text, label))
            if len(seen) >= max_per_service:
                break

    # reinforce homes and explicit cloud intents
    for cloud, label, extras in [
        ("azure", "azure:home", ["microsoft azure portal", "azure dashboard", "azure cloud"]),
        ("aws", "aws:home", ["amazon web services console", "aws dashboard", "aws management console"]),
        ("gcp", "gcp:home", ["google cloud console", "gcp dashboard", "google cloud platform"]),
    ]:
        for t in [f"open {cloud}", f"go to {cloud}", f"navigate to {cloud}", f"launch {cloud} console", f"{cloud} home"] + extras:
            for b in ["", " in firefox", " in chrome"]:
                examples.append((t + b, label))

    rnd.shuffle(examples)
    return examples
