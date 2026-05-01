"""Dataset version management utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _next_version(versions_dir: Path, ticker: str, start: str, end: str) -> int:
    """Compute next integer version for ticker/date-range dataset."""
    pattern = f"{ticker}_{start}_{end}_v*.csv"
    existing = sorted(versions_dir.glob(pattern))
    if not existing:
        return 1

    max_version = 0
    for item in existing:
        stem = item.stem
        if "_v" not in stem:
            continue
        try:
            version = int(stem.rsplit("_v", 1)[1])
        except ValueError:
            continue
        max_version = max(max_version, version)
    return max_version + 1


def _read_metadata(metadata_path: Path) -> dict:
    """Read metadata JSON with safe fallback."""
    if not metadata_path.exists():
        return {"datasets": []}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"datasets": []}
        if "datasets" not in data or not isinstance(data["datasets"], list):
            data["datasets"] = []
        return data
    except Exception:
        return {"datasets": []}


def save_versioned_dataset(
    df: pd.DataFrame,
    ticker: str,
    start: str,
    end: str,
    source: str,
    datasets_root: Path,
) -> dict:
    """Save dataset as versioned CSV and update metadata catalog."""
    versions_dir = datasets_root / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = datasets_root / "metadata.json"

    version = _next_version(versions_dir, ticker=ticker, start=start, end=end)
    version_name = f"{ticker}_{start}_{end}_v{version}.csv"
    version_path = versions_dir / version_name

    df.to_csv(version_path, index_label="Date")

    record = {
        "ticker": ticker,
        "start": start,
        "end": end,
        "source": source,
        "version": version,
        "rows": int(len(df)),
        "missing_ratio": float(df.isna().mean().mean()) if not df.empty else 1.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file": str(version_path.name),
    }

    metadata = _read_metadata(metadata_path)
    metadata["datasets"].append(record)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return record
