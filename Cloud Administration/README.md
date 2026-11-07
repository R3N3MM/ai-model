# Cloud Admin Browser Agent — AI model from scratch

A local cloud administration agent with an **AI intent model built and trained from scratch** (pure Python). It interprets user requests and navigates to **Microsoft Azure**, **AWS**, and **Google Cloud Platform (GCP)** using a browser you choose.

## AI model (from scratch, no downloads)
- **No pre-trained/downloaded model weights**: `cloud_admin_model.json` is generated locally by training on built-in examples.
- **Architecture**: bag-of-words vectorizer + a feedforward neural network (Dense/ReLU hidden layer → Softmax output) implemented in pure Python (`cloud_admin_agent/ai_model/`).
- **Training**: cross-entropy loss with SGD on a local dataset of cloud/browser navigation intents (`dataset.py`). No external ML libraries or data downloads required.
- **Inference**: the agent loads the local model to predict the target cloud/service from your request, with a deterministic rule-based fallback.

## Features
- **Multi-browser support**: Firefox, Chrome/Google, Edge, Brave, Opera (via Selenium when available) plus a standard-library fallback.
- **Cloud targets**: Azure Portal, AWS Console, GCP Console, and common service shortcuts (VMs/EC2/Compute Engine, storage/S3/buckets, IAM, Kubernetes, SQL, etc.).
- **Interactive mode**: type natural-ish requests like `open azure portal` or `go to aws s3`.

## Project layout
```text
Cloud Administration/
├── main.py                  # CLI entry point
├── cloud_admin_model.json   # Locally trained model weights (generated, not downloaded)
├── requirements.txt         # Optional dependencies (Selenium)
├── config.example.json      # Optional browser/driver config
├── README.md
└── cloud_admin_agent/
    ├── __init__.py
    ├── parser.py            # Local request parser (browser/intent helpers)
    ├── clouds.py            # Cloud destinations, shortcuts, model+rule resolution
    ├── browser.py           # Browser control (Selenium + webbrowser fallback)
    └── ai_model/            # AI model from scratch (pure Python)
        ├── dataset.py       # Built-in training examples
        ├── tokenizer.py     # Bag-of-words vectorizer
        ├── network.py       # MLP (forward/backprop/SGD) from scratch
        ├── train.py         # Local training script
        └── inference.py     # Load/predict using local weights
```

## Setup
Requires Python 3.9+ (standard library only for the AI model).

### Train / regenerate the AI model (from scratch)
The repository includes a pre-generated `cloud_admin_model.json`, but you can retrain it locally at any time (no downloads):
```bash
python3 -m cloud_admin_agent.ai_model.train
# optional: --epochs 400 --hidden 32 --lr 0.03 --out cloud_admin_model.json
```

### Basic (no extra packages)
The agent can open URLs using Python's built-in `webbrowser` module:
```bash
python3 main.py --browser firefox "open azure portal"
```

### Full browser automation (recommended)
Install Selenium to let the agent control a browser session:
```bash
python3 -m pip install -r requirements.txt
```

You also need a compatible browser + WebDriver available on your system (for example `geckodriver` for Firefox, or the appropriate driver for Chromium-based browsers). Driver setup varies by OS/browser.

## Usage
Interactive mode:
```bash
python3 main.py
```

Single request:
```bash
python3 main.py "go to gcp compute engine"
python3 main.py --browser brave "open aws iam"
python3 main.py --cloud azure "show virtual machines"
```

Options:
- `--browser {firefox,chrome,google,edge,brave,opera,auto}`: preferred browser.
- `--cloud {azure,aws,gcp,auto}`: hint which cloud to target.
- `--no-selenium`: force the built-in `webbrowser` fallback.

## Examples
- `open microsoft azure`
- `navigate to aws console`
- `go to google cloud platform`
- `open azure virtual machines`
- `show aws s3 buckets` (opens S3 console)
- `gcp iam`

## Notes
- **AI model from scratch**: the intent model is implemented and trained locally (pure Python). No pre-trained weights or external ML models are downloaded. The generated `cloud_admin_model.json` contains locally learned weights.
- This is a small, purpose-built model (cloud navigation intents), not a general-purpose LLM. It falls back to deterministic rules when needed.
- Opening cloud consoles may require you to sign in; the agent navigates to the requested destination but does not bypass authentication.
- "Google" in the browser list is treated as Google Chrome.
