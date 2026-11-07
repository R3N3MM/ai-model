"""Cloud destinations and service shortcuts.

Knowledge/skills are derived from official cloud documentation catalogs
(see SOURCES.md) and loaded from skills_data/catalog.json when available.
No pre-trained models are downloaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

CATALOG_PATH = Path(__file__).resolve().parents[1] / "skills_data" / "catalog.json"


@dataclass(frozen=True)
class Destination:
    cloud: str
    name: str
    url: str
    aliases: Tuple[str, ...] = ()
    key: str = ""
    category: str = ""


CLOUD_HOME = {
    "azure": Destination(
        cloud="azure",
        name="Microsoft Azure Portal",
        url="https://portal.azure.com/",
        aliases=("azure", "microsoft azure", "azure portal", "ms azure"),
    ),
    "aws": Destination(
        cloud="aws",
        name="AWS Management Console",
        url="https://console.aws.amazon.com/",
        aliases=("aws", "amazon web services", "aws console", "amazon aws"),
    ),
    "gcp": Destination(
        cloud="gcp",
        name="Google Cloud Console",
        url="https://console.cloud.google.com/",
        aliases=("gcp", "google cloud", "google cloud platform", "gcloud", "google cloud console"),
    ),
}

# Common service shortcuts. Add more as needed.
SERVICES = [
    # Azure
    Destination("azure", "Azure Virtual Machines", "https://portal.azure.com/#view/HubsExtension/BrowseResource/resourceType/Microsoft.Compute%2FVirtualMachines", ("vm", "vms", "virtual machine", "virtual machines", "compute")),
    Destination("azure", "Azure Storage Accounts", "https://portal.azure.com/#view/HubsExtension/BrowseResource/resourceType/Microsoft.Storage%2FStorageAccounts", ("storage", "blob", "storage account", "storage accounts")),
    Destination("azure", "Azure Active Directory / Entra ID", "https://portal.azure.com/#view/Microsoft_AAD_IAM/ActiveDirectoryMenuBlade/~/Overview", ("entra", "entra id", "active directory", "aad", "iam", "identity")),
    Destination("azure", "Azure Kubernetes Service", "https://portal.azure.com/#view/HubsExtension/BrowseResource/resourceType/Microsoft.ContainerService%2FmanagedClusters", ("aks", "kubernetes", "k8s")),
    Destination("azure", "Azure Resource Groups", "https://portal.azure.com/#view/HubsExtension/BrowseResourceGroups", ("resource group", "resource groups", "rg")),
    Destination("azure", "Azure SQL Databases", "https://portal.azure.com/#view/HubsExtension/BrowseResource/resourceType/Microsoft.Sql%2Fservers%2Fdatabases", ("sql", "database", "databases")),
    # AWS
    Destination("aws", "Amazon EC2", "https://console.aws.amazon.com/ec2/", ("ec2", "vm", "vms", "virtual machine", "virtual machines", "compute", "instances")),
    Destination("aws", "Amazon S3", "https://console.aws.amazon.com/s3/", ("s3", "storage", "bucket", "buckets", "object storage")),
    Destination("aws", "AWS IAM", "https://console.aws.amazon.com/iam/", ("iam", "identity", "users", "roles", "permissions")),
    Destination("aws", "Amazon EKS", "https://console.aws.amazon.com/eks/", ("eks", "kubernetes", "k8s")),
    Destination("aws", "AWS Lambda", "https://console.aws.amazon.com/lambda/", ("lambda", "functions", "serverless")),
    Destination("aws", "Amazon RDS", "https://console.aws.amazon.com/rds/", ("rds", "sql", "database", "databases")),
    # GCP
    Destination("gcp", "Compute Engine", "https://console.cloud.google.com/compute/instances", ("compute engine", "gce", "vm", "vms", "virtual machine", "virtual machines", "instances", "compute")),
    Destination("gcp", "Cloud Storage", "https://console.cloud.google.com/storage/browser", ("cloud storage", "gcs", "storage", "bucket", "buckets")),
    Destination("gcp", "IAM & Admin", "https://console.cloud.google.com/iam-admin/iam", ("iam", "identity", "permissions", "admin")),
    Destination("gcp", "Google Kubernetes Engine", "https://console.cloud.google.com/kubernetes/list/overview", ("gke", "kubernetes", "k8s")),
    Destination("gcp", "Cloud Functions", "https://console.cloud.google.com/functions/list", ("cloud functions", "functions", "serverless")),
    Destination("gcp", "Cloud SQL", "https://console.cloud.google.com/sql/instances", ("cloud sql", "sql", "database", "databases")),
]


def detect_cloud(text: str) -> Optional[str]:
    t = text.lower()
    # Prefer more specific phrases first.
    checks = [
        ("gcp", ("google cloud platform", "google cloud", "gcp", "gcloud")),
        ("azure", ("microsoft azure", "azure", "ms azure")),
        ("aws", ("amazon web services", "aws", "amazon")),
    ]
    for cloud, keys in checks:
        if any(k in t for k in keys):
            return cloud
    return None

@lru_cache(maxsize=1)
def load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        return []
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _dest_from_item(item: dict) -> Destination:
    return Destination(
        cloud=item["cloud"], name=item["name"], url=item["url"],
        aliases=tuple(item.get("aliases") or ()), key=item.get("key", ""),
        category=item.get("category", ""),
    )


def all_destinations() -> list[Destination]:
    items = load_catalog()
    if items:
        return [_dest_from_item(i) for i in items]
    return list(CLOUD_HOME.values()) + list(SERVICES)


def resolve_label(label: str) -> Optional[Destination]:
    """Resolve a model-predicted label like 'aws:s3' or 'azure:virtual-machines'."""
    if ":" not in label:
        return None
    cloud, key = label.split(":", 1)
    cloud = cloud.strip().lower()
    key = key.strip().lower()
    for item in load_catalog():
        if item.get("cloud") == cloud and item.get("key", "").lower() == key:
            return _dest_from_item(item)
    if key == "home":
        return CLOUD_HOME.get(cloud)
    for dest in SERVICES:
        if dest.cloud != cloud:
            continue
        if dest.name.lower() == key or key in (a.lower() for a in dest.aliases):
            return dest
    for dest in SERVICES:
        if dest.cloud == cloud and (key in dest.name.lower() or any(key in a.lower() for a in dest.aliases)):
            return dest
    return CLOUD_HOME.get(cloud)


def _best_match(text: str, cloud: Optional[str]) -> Optional[Destination]:
    t = " ".join(text.lower().split())
    best: Optional[Tuple[int, Destination]] = None
    for dest in all_destinations():
        if cloud and dest.cloud != cloud:
            continue
        candidates = [dest.name.lower(), dest.key.lower().replace('-', ' ')] + [a.lower() for a in dest.aliases]
        for alias in candidates:
            if not alias or len(alias) < 2:
                continue
            if alias in t:
                score = len(alias) + (10 if dest.key and alias == dest.key.lower().replace('-', ' ') else 0)
                if best is None or score > best[0]:
                    best = (score, dest)
    return best[1] if best else None


def resolve_destination(text: str, cloud_hint: Optional[str] = None, model_path: Optional[str] = None) -> Destination:
    """Resolve a user request to a cloud destination.

    Uses the locally trained from-scratch AI model (skills derived from
    official cloud doc catalogs) when available, then deterministic matching
    over the skills catalog. No pre-trained models are downloaded.
    """
    try:
        from .ai_model.inference import try_load

        model = try_load(model_path)
        if model is not None:
            # Prefer highest-confidence prediction that resolves to a known catalog destination.
            preds = model.predict_topk(text, k=5) if hasattr(model, "predict_topk") else [model.predict(text)]
            for label, confidence in preds:
                dest = resolve_label(label)
                if dest is not None and confidence >= 0.22:
                    return dest
            label, confidence = model.predict(text)
            dest = resolve_label(label)
            if dest is not None and confidence >= 0.18:
                return dest
    except Exception:
        pass

    t = " ".join(text.lower().split())
    cloud = cloud_hint or detect_cloud(t)
    match = _best_match(t, cloud)
    if match:
        return match
    if cloud:
        # catalog home
        for item in load_catalog():
            if item.get("cloud") == cloud and item.get("key") == "home":
                return _dest_from_item(item)
        return CLOUD_HOME[cloud]
    match = _best_match(t, None)
    if match:
        return match
    detected = detect_cloud(t)
    if detected:
        return CLOUD_HOME[detected]
    return CLOUD_HOME["azure"]

