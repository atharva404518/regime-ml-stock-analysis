"""Backward-compatible wrapper around the new ingestion loader."""

from __future__ import annotations

import pandas as pd

from data.loader import load_market_data


def load_ohlcv_data(
    ticker: str,
    start: str,
    end: str,
    offline: bool = False,
    refresh_data: bool = False,
    data_mode: str = "auto",
    max_retries: int = 5,
    missing_threshold: float = 0.20,
) -> pd.DataFrame:
    """Compatibility loader preserving previous function signature.

    Legacy mapping:
    - ``offline=True`` forces local mode.
    - ``refresh_data=True`` with auto mode prefers tiingo mode.
    - ``data_mode`` aliases ``api->tiingo`` and ``cache->local``.
    """
    mode = (data_mode or "auto").strip().lower()
    if mode == "api":
        mode = "tiingo"
    if mode == "cache":
        mode = "local"

    if offline:
        mode = "local"
    elif refresh_data and mode == "auto":
        mode = "tiingo"

    return load_market_data(
        ticker=ticker,
        start=start,
        end=end,
        mode=mode,
        missing_threshold=missing_threshold,
        max_retries=max_retries,
    )
