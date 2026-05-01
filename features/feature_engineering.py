"""Feature engineering utilities for quantitative research workflows."""

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create backward-looking technical features and forward return targets.

    Args:
        df: Cleaned OHLCV DataFrame with columns:
            ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

    Returns:
        DataFrame containing original columns, engineered features, and targets,
        with rows containing rolling-window or shift NaNs removed.

    Raises:
        ValueError: If required columns are missing or NaNs remain after cleaning.
    """
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    LOGGER.info("Starting feature engineering on %d rows.", len(df))

    features_df = df.copy()

    # 1) Log return based on Close.
    features_df["log_return"] = np.log(features_df["Close"] / features_df["Close"].shift(1))

    # 2) Rolling moving averages.
    features_df["ma_5"] = features_df["Close"].rolling(window=5, min_periods=5).mean()
    features_df["ma_20"] = features_df["Close"].rolling(window=20, min_periods=20).mean()
    features_df["ma_50"] = features_df["Close"].rolling(window=50, min_periods=50).mean()

    # 3) Rolling volatility (20-day std of log returns).
    features_df["volatility_20"] = (
        features_df["log_return"].rolling(window=20, min_periods=20).std()
    )

    # 4) Momentum (5-day absolute close change).
    features_df["momentum_5"] = features_df["Close"] - features_df["Close"].shift(5)

    # 5) RSI (14-day), implemented manually.
    delta = features_df["Close"].diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(window=14, min_periods=14).mean()
    avg_loss = losses.rolling(window=14, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    features_df["rsi_14"] = rsi

    # 6) Volume z-score (20-day).
    vol_mean_20 = features_df["Volume"].rolling(window=20, min_periods=20).mean()
    vol_std_20 = features_df["Volume"].rolling(window=20, min_periods=20).std()
    features_df["volume_zscore_20"] = (
        (features_df["Volume"] - vol_mean_20) / vol_std_20.replace(0.0, np.nan)
    )

    # Forward one-step target: next-day simple return.
    features_df["target"] = features_df["Close"].pct_change().shift(-1)
    # Forward 5-day target return aligned with today's features.
    features_df["target_5d"] = (features_df["Close"].shift(-5) / features_df["Close"]) - 1
    features_df["target_5d_class"] = (features_df["target_5d"] > 0).astype(int)

    # Strict anti-leakage: all engineered predictors are shifted by 1 day so
    # the feature at time t uses information available only up to t-1.
    predictor_cols = [
        "log_return",
        "ma_5",
        "ma_20",
        "ma_50",
        "volatility_20",
        "momentum_5",
        "rsi_14",
        "volume_zscore_20",
    ]
    features_df[predictor_cols] = features_df[predictor_cols].shift(1)

    initial_rows = len(features_df)
    features_df = features_df.dropna(axis=0).copy()
    dropped_rows = initial_rows - len(features_df)
    LOGGER.info(
        "Feature engineering complete. Rows before: %d, after cleaning: %d, dropped: %d.",
        initial_rows,
        len(features_df),
        dropped_rows,
    )

    if features_df.isna().any().any():
        raise ValueError("NaN values remain after feature engineering cleanup.")

    return features_df
