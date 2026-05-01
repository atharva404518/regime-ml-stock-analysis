"""Data quality report generation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_quality_report(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Generate a structured quality report for OHLCV dataframe."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "rows": 0,
            "missing_pct": 100.0,
            "duplicates": 0,
            "outliers": 0,
            "verdict": "fail",
        }

    rows = int(len(df))
    missing_pct = float(df.isna().mean().mean() * 100.0)
    duplicates = int(df.duplicated().sum() + df.index.duplicated().sum())

    close = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce")
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty or float(returns.std(ddof=0)) == 0.0:
        outliers = 0
    else:
        z = (returns - returns.mean()) / returns.std(ddof=0)
        outliers = int((z.abs() > 3.0).sum())

    verdict = "pass"
    if rows < 30 or missing_pct > 5.0 or duplicates > 0:
        verdict = "warn"
    if rows < 10 or missing_pct > 20.0:
        verdict = "fail"

    return {
        "rows": rows,
        "missing_pct": round(missing_pct, 4),
        "duplicates": duplicates,
        "outliers": outliers,
        "verdict": verdict,
    }
