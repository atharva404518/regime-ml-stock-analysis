"""Market regime labeling utilities for quantitative pipelines."""

import numpy as np
import pandas as pd


def add_market_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend, volatility, and crash regime labels to market data.

    Args:
        df: Input DataFrame containing at least a ``Close`` column.

    Returns:
        A new DataFrame with regime columns:
        ``Trend_Regime``, ``Vol_Regime``, and ``Crash_Flag``.

    Raises:
        ValueError: If required columns are missing or percentile inputs are empty.
    """
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    regime_df = df.copy()

    # Trend regime components based only on current/past prices.
    regime_df["SMA50"] = regime_df["Close"].rolling(window=50, min_periods=50).mean()
    regime_df["SMA200"] = regime_df["Close"].rolling(window=200, min_periods=200).mean()
    regime_df["Slope50"] = regime_df["SMA50"] - regime_df["SMA50"].shift(5)

    bull_mask = (
        (regime_df["Close"] > regime_df["SMA200"])
        & (regime_df["SMA50"] > regime_df["SMA200"])
        & (regime_df["Slope50"] > 0)
    )
    bear_mask = (
        (regime_df["Close"] < regime_df["SMA200"])
        & (regime_df["SMA50"] < regime_df["SMA200"])
        & (regime_df["Slope50"] < 0)
    )

    regime_df["Trend_Regime"] = "Sideways"
    regime_df.loc[bull_mask, "Trend_Regime"] = "Bull"
    regime_df.loc[bear_mask, "Trend_Regime"] = "Bear"

    # Volatility regime from rolling historical daily returns.
    regime_df["Daily_Return"] = regime_df["Close"].pct_change()
    regime_df["RollingVol20"] = regime_df["Daily_Return"].rolling(
        window=20, min_periods=20
    ).std()

    vol_series = regime_df["RollingVol20"].dropna()
    if vol_series.empty:
        raise ValueError("Not enough data to compute RollingVol20 percentiles.")

    vol_p25 = vol_series.quantile(0.25)
    vol_p75 = vol_series.quantile(0.75)

    regime_df["Vol_Regime"] = "NormalVol"
    regime_df.loc[regime_df["RollingVol20"] > vol_p75, "Vol_Regime"] = "HighVol"
    regime_df.loc[regime_df["RollingVol20"] < vol_p25, "Vol_Regime"] = "LowVol"

    # Crash flag: crash day and next 5 trading days.
    crash_day = (regime_df["Daily_Return"] < -0.05).fillna(False).astype(int).to_numpy()
    crash_window = np.convolve(crash_day, np.ones(6, dtype=int), mode="full")[
        : len(regime_df)
    ]
    regime_df["Crash_Flag"] = (crash_window > 0).astype(int)

    # Required cleanup for downstream modeling.
    regime_df = regime_df.dropna(subset=["SMA200"]).copy()
    return regime_df
