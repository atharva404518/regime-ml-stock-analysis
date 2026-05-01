"""Regime-wise evaluation metrics for model predictions."""

import numpy as np
import pandas as pd


def _aggregate_metrics_by_group(
    eval_df: pd.DataFrame,
    group_col: str,
    regime_type: str,
    model_name: str,
) -> pd.DataFrame:
    """Aggregate error and directional metrics by a regime column."""
    grouped = (
        eval_df.groupby(group_col, dropna=False)
        .agg(
            Count=("y_true", "size"),
            RMSE=("error_sq", lambda s: float(np.sqrt(s.mean()))),
            MAE=("abs_error", "mean"),
            Directional_Accuracy=("dir_match", "mean"),
        )
        .reset_index()
        .rename(columns={group_col: "Regime_Value"})
    )
    grouped["Model"] = model_name
    grouped["Regime_Type"] = regime_type
    return grouped[
        [
            "Model",
            "Regime_Type",
            "Regime_Value",
            "Count",
            "RMSE",
            "MAE",
            "Directional_Accuracy",
        ]
    ]


def evaluate_by_regime(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate predictions by market regime on aligned test-set data.

    Args:
        df: Test-set DataFrame containing ``Trend_Regime``, ``Vol_Regime``,
            and ``Crash_Flag``.
        y_true: Ground-truth values aligned to the test-set index.
        y_pred: Model predictions aligned to the test-set index.
        model_name: Label used in the output ``Model`` column.

    Returns:
        Concatenated DataFrame with columns:
        ``["Model", "Regime_Type", "Regime_Value", "Count", "RMSE", "MAE",
        "Directional_Accuracy"]``.

    Raises:
        ValueError: If required regime columns are missing or aligned data is empty.
    """
    required_cols = {"Trend_Regime", "Vol_Regime", "Crash_Flag"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required regime columns: {sorted(missing_cols)}")

    eval_df = df.copy()
    aligned_targets = pd.concat(
        [y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1, join="inner"
    ).dropna()
    if aligned_targets.empty:
        raise ValueError("Aligned y_true/y_pred data is empty after joining.")

    eval_df = eval_df.loc[aligned_targets.index].copy()
    eval_df["y_true"] = aligned_targets["y_true"]
    eval_df["y_pred"] = aligned_targets["y_pred"]

    eval_df["error_sq"] = (eval_df["y_true"] - eval_df["y_pred"]) ** 2
    eval_df["abs_error"] = (eval_df["y_true"] - eval_df["y_pred"]).abs()
    eval_df["dir_match"] = (
        np.sign(eval_df["y_true"]) == np.sign(eval_df["y_pred"])
    ).astype(float)
    eval_df["Crash_Regime"] = np.where(eval_df["Crash_Flag"] == 1, "Crash", "Non-Crash")

    trend_metrics = _aggregate_metrics_by_group(
        eval_df=eval_df,
        group_col="Trend_Regime",
        regime_type="Trend",
        model_name=model_name,
    )
    vol_metrics = _aggregate_metrics_by_group(
        eval_df=eval_df,
        group_col="Vol_Regime",
        regime_type="Volatility",
        model_name=model_name,
    )
    crash_metrics = _aggregate_metrics_by_group(
        eval_df=eval_df,
        group_col="Crash_Regime",
        regime_type="Crash",
        model_name=model_name,
    )

    return pd.concat([trend_metrics, vol_metrics, crash_metrics], ignore_index=True)
