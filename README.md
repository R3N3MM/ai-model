# AI Model

This repository contains local AI/model projects.

## Projects

### Cloud Administration (`Cloud Administration/`)
A **from-scratch** cloud administration browser agent (no pre-trained/downloaded ML model weights). It uses a locally implemented and trainable intent model (pure Python: BoW + bigrams + MLP) plus a skills catalog derived from official Azure/AWS/GCP documentation catalogs to navigate to **Microsoft Azure**, **AWS**, and **Google Cloud Platform (GCP)** in browsers (Firefox, Chrome/Google, Edge, Brave, Opera).

- Skills (researched from official online catalogs): `Cloud Administration/skills/skills.md`
- Sources: `Cloud Administration/SOURCES.md`
- Usage/training: `Cloud Administration/README.md`

> The on-disk `cloud_admin_model.json` may need retraining to cover the full expanded skills catalog (training is user-controlled; see the project README).

### Cybersecurity (`Cybersecurity/`)
Folder created for future cybersecurity models/projects.

## Getting started
```bash
cd "Cloud Administration"
python3 main.py --no-selenium "open aws iam"
# Retrain the local from-scratch model only when you choose to:
python3 -m cloud_admin_agent.ai_model.train
```

## For Juandre
Clone or download this repository (see GitHub/GitLab download options, or `git clone <repo-url>`). No pre-trained ML models are downloaded by the project.
