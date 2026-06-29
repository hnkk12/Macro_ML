import json
import os
from typing import Dict, Any

METADATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "raw", "metadata.json"
)

def load_metadata() -> Dict[str, Any]:
    """Load metadata from data/raw/metadata.json."""
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_metadata(metadata: Dict[str, Any]) -> None:
    """Save metadata to data/raw/metadata.json."""
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def update_series_metadata(
    series_id: str,
    source: str,
    frequency: str,
    download_date: str,
    start_date: str,
    end_date: str,
    transformation: str
) -> None:
    """Update metadata for a specific series."""
    meta = load_metadata()
    meta[series_id] = {
        "series_id": series_id,
        "source": source,
        "frequency": frequency,
        "download_date": download_date,
        "start_date": start_date,
        "end_date": end_date,
        "transformation": transformation
    }
    save_metadata(meta)
