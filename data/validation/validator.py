"""Validation utilities for OHLCV datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class ValidationSummary:
    rows: int
    missing_ratio: float
    duplicate_rows: int
    duplicate_index: int
    continuity_ratio: float


def validate_ohlcv_dataframe(
    df: pd.DataFrame,
    missing_threshold: float = 0.20,
    enforce_continuity: bool = False,
) -> pd.DataFrame:
    """Validate and clean OHLCV DataFrame for downstream modeling."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    out = df.copy()
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()
    out = out.sort_index()

    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    out = out[~out.index.duplicated(keep="first")]

    out = out[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")

    raw_missing_ratio = float(out.isna().mean().mean())
    if raw_missing_ratio > missing_threshold:
        raise ValueError(
            f"Missing ratio {raw_missing_ratio:.2%} exceeds threshold {missing_threshold:.2%}."
        )

    out = out.ffill().bfill()

    remaining_missing = out.isna().mean()
    if bool((remaining_missing > 0).any()):
        missing_names = remaining_missing[remaining_missing > 0].index.tolist()
        raise ValueError(f"Remaining missing values after fill in: {missing_names}")

    if enforce_continuity:
        expected = pd.date_range(start=out.index.min(), end=out.index.max(), freq="B")
        continuity_ratio = float(len(out.index.intersection(expected)) / len(expected)) if len(expected) > 0 else 1.0
        if continuity_ratio < 0.90:
            raise ValueError(f"Date continuity too low: {continuity_ratio:.2%}")

    if out.empty:
        raise ValueError("Dataframe is empty after validation.")

    return out


def summarize_validation(df: pd.DataFrame) -> ValidationSummary:
    """Build compact validation summary metrics."""
    if df.empty:
        return ValidationSummary(0, 1.0, 0, 0, 0.0)

    expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
    continuity_ratio = float(len(df.index.intersection(expected)) / len(expected)) if len(expected) > 0 else 1.0
    return ValidationSummary(
        rows=len(df),
        missing_ratio=float(df.isna().mean().mean()),
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_index=int(df.index.duplicated().sum()),
        continuity_ratio=continuity_ratio,
    )
