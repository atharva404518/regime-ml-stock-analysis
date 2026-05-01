"""Local CSV source adapter for OHLCV ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_local_csv_ohlcv(
    ticker: str,
    start: str,
    end: str,
    base_dir: Path,
) -> pd.DataFrame:
    """Load OHLCV from local CSV file in datasets/raw directory."""
    ticker = ticker.strip().upper()
    candidate_paths = [
        base_dir / f"{ticker}.csv",
        Path(__file__).resolve().parents[1] / "local" / f"{ticker}.csv",
    ]
    file_path = next((path for path in candidate_paths if path.exists()), None)
    if file_path is None:
        raise FileNotFoundError(f"Local CSV not found for {ticker} in known locations.")

    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Local CSV is empty: {file_path}")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[~df["Date"].isna()].copy()
        df = df.set_index("Date")
    else:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df[~df[first_col].isna()].copy()
        df = df.set_index(first_col)

    df = df.sort_index()
    df = df.loc[pd.to_datetime(start) : pd.to_datetime(end)]
    if df.empty:
        raise ValueError(f"Local CSV has no rows in date range {start} to {end}: {file_path}")

    return df
