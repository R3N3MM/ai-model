"""Browser control for the cloud admin agent.

Supports Selenium-driven sessions when available, with a built-in
`webbrowser` fallback. No AI models are downloaded.
"""

from __future__ import annotations

import shutil
import webbrowser
from dataclasses import dataclass
from typing import Optional


SUPPORTED = ("auto", "firefox", "chrome", "google", "edge", "brave", "opera")


@dataclass
class BrowserConfig:
    browser: str = "auto"
    prefer_selenium: bool = True
    browser_path: Optional[str] = None
    driver_path: Optional[str] = None


def _normalize(browser: Optional[str]) -> str:
    b = (browser or "auto").lower().strip()
    if b == "google chrome":
        return "chrome"
    if b == "microsoft edge":
        return "edge"
    return b if b in SUPPORTED else "auto"


def _detect_browser() -> str:
    for name, cmds in (
        ("firefox", ("firefox",)),
        ("brave", ("brave-browser", "brave")),
        ("chrome", ("google-chrome", "google-chrome-stable", "chrome", "chromium-browser", "chromium")),
        ("edge", ("microsoft-edge", "microsoft-edge-stable", "msedge")),
        ("opera", ("opera",)),
    ):
        if any(shutil.which(c) for c in cmds):
            return name
    return "firefox"


def open_url(url: str, config: Optional[BrowserConfig] = None) -> str:
    """Open a URL in the requested browser. Returns a status message."""
    cfg = config or BrowserConfig()
    browser = _normalize(cfg.browser)
    if browser == "auto":
        browser = _detect_browser()
    if browser == "google":
        browser = "chrome"

    if cfg.prefer_selenium:
        msg = _open_with_selenium(url, browser, cfg)
        if msg:
            return msg

    return _open_with_webbrowser(url, browser, cfg)


def _open_with_webbrowser(url: str, browser: str, cfg: BrowserConfig) -> str:
    try:
        # Try a specific browser name; fall back to system default.
        name_map = {
            "firefox": "firefox",
            "chrome": "google-chrome",
            "edge": "microsoft-edge",
            "brave": "brave-browser",
            "opera": "opera",
        }
        target = cfg.browser_path or name_map.get(browser)
        if target:
            try:
                ctl = webbrowser.get(target)
                ctl.open_new_tab(url)
                return f"Opened {url} in {browser} (webbrowser fallback)."
            except Exception:
                pass
        webbrowser.open_new_tab(url)
        return f"Opened {url} in the system default browser (fallback)."
    except Exception as exc:
        return f"Failed to open browser: {exc}. URL: {url}"


def _open_with_selenium(url: str, browser: str, cfg: BrowserConfig) -> Optional[str]:
    try:
        from selenium import webdriver  # type: ignore
    except Exception:
        return None

    try:
        driver = _make_driver(webdriver, browser, cfg)
    except Exception as exc:
        return f"Selenium could not start {browser}: {exc}. Try --no-selenium or configure a driver."

    try:
        driver.get(url)
        return f"Navigated to {url} in {browser} via Selenium (session left open)."
    except Exception as exc:
        try:
            driver.quit()
        except Exception:
            pass
        return f"Selenium navigation failed: {exc}. URL: {url}"


def _make_driver(webdriver, browser: str, cfg: BrowserConfig):
    # Firefox
    if browser == "firefox":
        from selenium.webdriver.firefox.options import Options  # type: ignore
        from selenium.webdriver.firefox.service import Service  # type: ignore

        opts = Options()
        if cfg.browser_path:
            opts.binary_location = cfg.browser_path
        service = Service(executable_path=cfg.driver_path) if cfg.driver_path else Service()
        return webdriver.Firefox(options=opts, service=service)

    # Chromium-based: chrome/google, edge, brave, opera (best-effort)
    if browser in ("chrome", "edge", "brave", "opera"):
        if browser == "edge":
            from selenium.webdriver.edge.options import Options  # type: ignore
            from selenium.webdriver.edge.service import Service  # type: ignore

            opts = Options()
            if cfg.browser_path:
                opts.binary_location = cfg.browser_path
            service = Service(executable_path=cfg.driver_path) if cfg.driver_path else Service()
            return webdriver.Edge(options=opts, service=service)

        from selenium.webdriver.chrome.options import Options  # type: ignore
        from selenium.webdriver.chrome.service import Service  # type: ignore

        opts = Options()
        if cfg.browser_path:
            opts.binary_location = cfg.browser_path
        # Brave/Opera often work as Chromium with a custom binary; driver compatibility varies.
        service = Service(executable_path=cfg.driver_path) if cfg.driver_path else Service()
        return webdriver.Chrome(options=opts, service=service)

    # Default: try Firefox, then Chrome.
    try:
        return _make_driver(webdriver, "firefox", cfg)
    except Exception:
        return _make_driver(webdriver, "chrome", cfg)
