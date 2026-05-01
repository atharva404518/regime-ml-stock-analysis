"""Bootstrap local fallback datasets from cached OHLCV files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _cache_path(project_root: Path, ticker: str, start: str, end: str) -> Path:
    return project_root / "data_cache" / f"{ticker}_{start}_{end}.csv"


def _local_path(project_root: Path, ticker: str) -> Path:
    return project_root / "data_local" / f"{ticker}.csv"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap local dataset from cache file.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    project_root = Path(__file__).resolve().parents[1]
    cache_file = _cache_path(project_root, ticker=ticker, start=args.start, end=args.end)
    local_file = _local_path(project_root, ticker=ticker)

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    df = pd.read_csv(cache_file)
    if df.empty:
        raise ValueError(f"Cache file is empty: {cache_file}")

    local_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(local_file, index=False)
    LOGGER.info("Saved local fallback dataset: %s", local_file)


if __name__ == "__main__":
    main()
