# Knowledge Sources (fetched online, thorough research)

The skills/knowledge base and training data are derived from **official cloud provider documentation and product catalogs fetched online**. No pre-trained models were downloaded.

| Provider | Source URL | Purpose |
|---|---|---|
| Microsoft Azure | https://learn.microsoft.com/en-us/azure/?product=popular | Azure documentation hub / product catalog |
| Microsoft Azure | https://azure.microsoft.com/en-us/products/ | Azure products catalog (fetched for research/coverage) |
| Microsoft Azure | https://learn.microsoft.com/en-us/azure/architecture/guide/technology-choices/technology-choices-overview | Architecture/technology choice guidance (research) |
| Amazon Web Services | https://docs.aws.amazon.com/ | AWS documentation home / service catalog |
| Amazon Web Services | https://aws.amazon.com/products/ | AWS products page (fetched for research) |
| Amazon Web Services | https://docs.aws.amazon.com/whitepapers/latest/aws-overview/introduction.html | AWS overview whitepaper (research) |
| Google Cloud | https://cloud.google.com/products | Google Cloud products catalog |
| Google Cloud | https://cloud.google.com/docs | Google Cloud documentation (reference) |

## How sources are used
- Service names, categories, and administration concepts were researched from the catalogs above.
- The curated skills catalog (`skills_data/catalog.json`, 186 entries) maps services to **official console URLs**, aliases, and admin skill phrases for high-accuracy navigation.
- Training examples are generated locally from this catalog for the from-scratch intent model (no external data/model downloads at runtime).

## Scope / honesty
- "All online documents" from every cloud giant is **not** exhaustively ingested (catalogs are extremely large and change). This build performs thorough research over the key official catalogs/pages above and produces a broad, admin-focused skills set.
- Console URLs point to official portals and may require authentication.
- To refresh: re-fetch sources, update `build_skills.py`/catalog, regenerate `skills/*.md`, and retrain the local model (user-controlled; training is not run automatically).
