"""Simple local request parser (no downloaded ML models).

Extracts intent (navigate/open/show), browser preference, and cloud hint
from free-form user text using deterministic rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


NAV_WORDS = (
    "open",
    "go to",
    "goto",
    "navigate",
    "navigate to",
    "show",
    "show me",
    "take me to",
    "visit",
    "launch",
    "access",
    "browse",
    "find",
    "view",
)

BROWSER_ALIASES = {
    "firefox": ("firefox", "mozilla"),
    "chrome": ("chrome", "google chrome"),
    "google": ("google",),  # per request: "Google" browser -> Chrome
    "edge": ("edge", "microsoft edge", "msedge"),
    "brave": ("brave",),
    "opera": ("opera",),
}


@dataclass
class ParsedRequest:
    raw: str
    intent: str
    browser: Optional[str]
    cloud: Optional[str]
    target_text: str


def parse(text: str, default_browser: Optional[str] = None, cloud_hint: Optional[str] = None) -> ParsedRequest:
    raw = text.strip()
    t = raw.lower()

    browser = default_browser
    for name, aliases in BROWSER_ALIASES.items():
        if any(a in t for a in aliases):
            browser = name
            break

    intent = "navigate" if any(w in t for w in NAV_WORDS) else "navigate"

    from .clouds import detect_cloud

    cloud = cloud_hint or detect_cloud(t)

    # Keep target text as-is for destination resolution; it's already descriptive.
    return ParsedRequest(raw=raw, intent=intent, browser=browser, cloud=cloud, target_text=raw)
