"""Synthetic OHLCV source for fail-safe ingestion fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate geometric-random-walk OHLCV data."""
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")

    if pd.isna(start_ts) or pd.isna(end_ts):
        end_ts = pd.Timestamp.today().normalize()
        start_ts = end_ts - pd.Timedelta(days=365)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    dates = pd.date_range(start=start_ts, end=end_ts, freq="B")
    if len(dates) < 30:
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=252, freq="B")

    np.random.seed(seed)
    n = len(dates)

    drift = 0.0003
    vol = 0.012
    log_returns = np.random.normal(loc=drift, scale=vol, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    open_noise = np.random.normal(0.0, 0.003, size=n)
    open_px = close * (1.0 + open_noise)
    if n > 1:
        open_px[1:] = close[:-1] * (1.0 + open_noise[1:])

    spread = np.abs(np.random.normal(0.0025, 0.0015, size=n))
    high = np.maximum(open_px, close) * (1.0 + spread)
    low = np.minimum(open_px, close) * (1.0 - spread)
    low = np.maximum(low, 0.01)

    volume = np.random.lognormal(mean=np.log(1_300_000.0), sigma=0.35, size=n)
    volume = np.maximum(volume, 1_000.0)

    df = pd.DataFrame(
        {
            "Open": open_px.astype(float),
            "High": high.astype(float),
            "Low": low.astype(float),
            "Close": close.astype(float),
            "Volume": volume.astype(float),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df
