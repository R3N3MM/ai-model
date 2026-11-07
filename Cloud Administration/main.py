#!/usr/bin/env python3
"""Cloud Admin Browser Agent - from scratch, no downloaded ML models.

Usage:
  python3 main.py
  python3 main.py "open azure portal"
  python3 main.py --browser firefox --cloud aws "go to s3"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cloud_admin_agent.browser import BrowserConfig, open_url
from cloud_admin_agent.clouds import resolve_destination
from cloud_admin_agent.parser import parse


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def handle_request(text: str, *, browser: str, cloud: str, no_selenium: bool, config: dict, model_path: str | None = None) -> str:
    req = parse(text, default_browser=None if browser == "auto" else browser, cloud_hint=None if cloud == "auto" else cloud)
    chosen_browser = req.browser or browser or config.get("default_browser") or "auto"
    destination = resolve_destination(req.target_text, cloud_hint=req.cloud, model_path=model_path)

    model_info = "rules (catalog + deterministic matching)"
    try:
        from cloud_admin_agent.ai_model.inference import try_load
        from cloud_admin_agent.clouds import load_catalog

        m = try_load(model_path)
        if m is not None:
            preds = m.predict_topk(req.target_text, k=3) if hasattr(m, "predict_topk") else [m.predict(req.target_text)]
            top = ", ".join(f"{l} {c:.2f}" for l, c in preds)
            cat_n = len(load_catalog())
            lbl_n = len(getattr(m, "labels", []) or [])
            stale = f" | NOTE: model has {lbl_n} labels but catalog has {cat_n} skills; retrain to improve accuracy" if cat_n and lbl_n and lbl_n < cat_n else ""
            model_info = f"from-scratch local MLP (top: {top}){stale}"
    except Exception:
        pass

    browser_paths = config.get("browser_paths") or {}
    driver_paths = config.get("driver_paths") or {}
    cfg = BrowserConfig(
        browser=chosen_browser,
        prefer_selenium=not no_selenium and bool(config.get("prefer_selenium", True)),
        browser_path=browser_paths.get(chosen_browser) or None,
        driver_path=driver_paths.get(chosen_browser) or None,
    )

    result = open_url(destination.url, cfg)
    return (
        f"Request: {req.raw}\n"
        f"AI: {model_info}\n"
        f"Interpreted: intent={req.intent}, cloud={destination.cloud}, browser={chosen_browser}\n"
        f"Destination: {destination.name}\n"
        f"URL: {destination.url}\n"
        f"Result: {result}"
    )


def interactive(args, config: dict) -> int:
    print("Cloud Admin Browser Agent (from-scratch local AI model, no downloaded ML models)")
    print("Targets: Microsoft Azure, AWS, Google Cloud Platform (GCP)")
    print("Browsers: Firefox, Chrome/Google, Edge, Brave, Opera")
    print("Type a request (e.g., 'open aws s3') or 'quit'.\n")
    while True:
        try:
            text = input("cloud-admin> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            return 0
        print(handle_request(text, browser=args.browser, cloud=args.cloud, no_selenium=args.no_selenium, config=config, model_path=args.model))
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="From-scratch cloud admin browser agent (local AI model, no downloaded ML models).")
    p.add_argument("request", nargs="*", help="User request, e.g. 'open azure portal'")
    p.add_argument("--browser", default="auto", choices=["auto", "firefox", "chrome", "google", "edge", "brave", "opera"], help="Preferred browser")
    p.add_argument("--cloud", default="auto", choices=["auto", "azure", "aws", "gcp"], help="Cloud hint")
    p.add_argument("--no-selenium", action="store_true", help="Use Python webbrowser fallback instead of Selenium")
    p.add_argument("--config", help="Path to JSON config (see config.example.json)")
    p.add_argument("--model", help="Path to the local from-scratch model JSON (default: cloud_admin_model.json)")
    args = p.parse_args(argv)
    config = load_config(args.config)

    if args.request:
        text = " ".join(args.request)
        print(handle_request(text, browser=args.browser, cloud=args.cloud, no_selenium=args.no_selenium, config=config, model_path=args.model))
        return 0
    return interactive(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
