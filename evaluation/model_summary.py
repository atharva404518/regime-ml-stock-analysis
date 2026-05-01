"""Model comparison summary architecture for quantitative evaluation outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from evaluation.scoring import compute_composite_score


def _to_float(value: Any) -> float:
    """Convert a value to float with NaN fallback."""
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_metric_lookup(metrics: dict[str, Any], keys: list[str]) -> float:
    """Fetch the first available metric from a dictionary."""
    for key in keys:
        if key in metrics:
            return _to_float(metrics[key])
    return float("nan")


def _to_dataframe(value: Any) -> pd.DataFrame:
    """Return a DataFrame or an empty DataFrame for non-DataFrame inputs."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame()


def _series_mean_std(df: pd.DataFrame, column: str) -> tuple[float, float]:
    """Compute mean and std for a DataFrame column with NaN-safe fallback."""
    if column not in df.columns:
        return float("nan"), float("nan")
    series = pd.to_numeric(df[column], errors="coerce")
    if not series.notna().any():
        return float("nan"), float("nan")
    return float(series.mean()), float(series.std(ddof=0))


def _regime_dispersion(regime_df: pd.DataFrame) -> float:
    """Compute regime dispersion as std of directional accuracy across regimes."""
    if "Directional_Accuracy" not in regime_df.columns:
        return float("nan")
    series = pd.to_numeric(regime_df["Directional_Accuracy"], errors="coerce")
    if not series.notna().any():
        return float("nan")
    return float(series.std(ddof=0))


def _normalize_series(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Min-max normalize a series while preserving NaN positions."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    min_val = float(valid.min())
    max_val = float(valid.max())
    if np.isclose(min_val, max_val):
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[valid.index] = 0.5
        return out
    scaled = (numeric - min_val) / (max_val - min_val)
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.astype(float)


def _compute_advanced_composite_score(df: pd.DataFrame) -> pd.Series:
    """Compute optional advanced composite score with requested weighted components."""
    sharpe_norm = _normalize_series(df["Sharpe"], higher_is_better=True) if "Sharpe" in df else pd.Series(np.nan, index=df.index)
    mean_ic_norm = _normalize_series(df["Mean_IC"], higher_is_better=True) if "Mean_IC" in df else pd.Series(np.nan, index=df.index)
    ic_t_norm = _normalize_series(df["IC_t_stat"], higher_is_better=True) if "IC_t_stat" in df else pd.Series(np.nan, index=df.index)
    da_norm = _normalize_series(df["Directional_Accuracy"], higher_is_better=True) if "Directional_Accuracy" in df else pd.Series(np.nan, index=df.index)
    max_dd_penalty = _normalize_series(df["Max_Drawdown_Abs"], higher_is_better=True) if "Max_Drawdown_Abs" in df else pd.Series(np.nan, index=df.index)
    sharpe_ci_positive = pd.to_numeric(df.get("Sharpe_CI_Lower", pd.Series(np.nan, index=df.index)), errors="coerce")
    sharpe_ci_positive = (sharpe_ci_positive > 0).astype(float)
    sharpe_ci_positive = sharpe_ci_positive.where(~pd.to_numeric(df.get("Sharpe_CI_Lower", pd.Series(np.nan, index=df.index)), errors="coerce").isna(), np.nan)

    component_map = {
        "Sharpe": (sharpe_norm, 0.30),
        "Mean_IC": (mean_ic_norm, 0.20),
        "IC_t_stat": (ic_t_norm, 0.15),
        "Directional_Accuracy": (da_norm, 0.15),
        "Sharpe_CI_Positive": (sharpe_ci_positive, 0.10),
        "Max_Drawdown": (max_dd_penalty, 0.10),
    }

    component_df = pd.DataFrame(
        {name: series for name, (series, _weight) in component_map.items()},
        index=df.index,
    )
    weights = pd.Series({name: weight for name, (_series, weight) in component_map.items()}, dtype=float)
    available_components = component_df.notna().any(axis=0)
    if not available_components.any():
        return pd.Series(np.nan, index=df.index, dtype=float)

    weights = weights[available_components]
    component_df = component_df[weights.index]
    weights = weights / weights.sum()

    weighted_sum = component_df.mul(weights, axis=1).sum(axis=1, skipna=True)
    present_weight_sum = component_df.notna().mul(weights, axis=1).sum(axis=1)
    return (weighted_sum / present_weight_sum.where(present_weight_sum > 0, np.nan)).astype(float)


def _extract_model_row(model_eval: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable metrics from a model evaluation payload."""
    model_name = str(model_eval.get("model_name", "Unknown_Model"))
    task_type = str(model_eval.get("task_type", "unknown")).lower()

    overall_metrics = model_eval.get("overall_metrics")
    trading_metrics = model_eval.get("trading_metrics")
    rolling_df = _to_dataframe(model_eval.get("rolling_metrics"))
    regime_df = _to_dataframe(model_eval.get("regime_metrics"))
    advanced_eval = model_eval.get("advanced_evaluation")

    overall_metrics = overall_metrics if isinstance(overall_metrics, dict) else {}
    trading_metrics = trading_metrics if isinstance(trading_metrics, dict) else {}
    advanced_eval = advanced_eval if isinstance(advanced_eval, dict) else {}
    ic_stats = advanced_eval.get("ic_stats") if isinstance(advanced_eval.get("ic_stats"), dict) else {}
    bootstrap_sharpe = (
        advanced_eval.get("bootstrap_sharpe")
        if isinstance(advanced_eval.get("bootstrap_sharpe"), dict)
        else {}
    )

    mse = _safe_metric_lookup(overall_metrics, ["mse", "model_mse"])
    rmse = _safe_metric_lookup(overall_metrics, ["rmse", "model_rmse"])

    directional_accuracy = _safe_metric_lookup(overall_metrics, ["directional_accuracy"])
    if np.isnan(directional_accuracy) and "Rolling_Directional_Accuracy" in rolling_df.columns:
        directional_accuracy = _to_float(
            pd.to_numeric(rolling_df["Rolling_Directional_Accuracy"], errors="coerce").mean()
        )

    mean_ic, ic_std = _series_mean_std(rolling_df, "Rolling_IC")

    # Required derived classification stats.
    mean_rolling_accuracy, rolling_accuracy_std = _series_mean_std(
        rolling_df,
        "Rolling_Accuracy",
    )
    classification_accuracy = _safe_metric_lookup(overall_metrics, ["accuracy"])
    if np.isnan(classification_accuracy):
        classification_accuracy = mean_rolling_accuracy

    sharpe = _safe_metric_lookup(trading_metrics, ["sharpe", "strategy_sharpe"])
    cumulative_return = _safe_metric_lookup(
        trading_metrics,
        ["cumulative_return", "strategy_cumulative_return"],
    )
    max_drawdown = _safe_metric_lookup(trading_metrics, ["max_drawdown"])
    max_drawdown_abs = abs(max_drawdown) if not np.isnan(max_drawdown) else float("nan")
    regime_dispersion = _regime_dispersion(regime_df)
    ic_t_stat = _safe_metric_lookup(ic_stats, ["ic_t_stat"])
    ic_p_value = _safe_metric_lookup(ic_stats, ["ic_p_value"])
    sharpe_ci_median = _safe_metric_lookup(bootstrap_sharpe, ["sharpe_median"])
    sharpe_ci_lower = _safe_metric_lookup(bootstrap_sharpe, ["sharpe_lower"])
    sharpe_ci_upper = _safe_metric_lookup(bootstrap_sharpe, ["sharpe_upper"])
    quantile_spread = _safe_metric_lookup(advanced_eval, ["quantile_spread"])

    return {
        "Model": model_name,
        "Task_Type": task_type,
        "MSE": mse,
        "RMSE": rmse,
        "Sharpe": sharpe,
        "Cumulative_Return": cumulative_return,
        "Mean_IC": mean_ic,
        "IC_Std": ic_std,
        "IC_t_stat": ic_t_stat,
        "IC_p_value": ic_p_value,
        "Sharpe_CI_Median": sharpe_ci_median,
        "Sharpe_CI_Lower": sharpe_ci_lower,
        "Sharpe_CI_Upper": sharpe_ci_upper,
        "Quantile_Spread": quantile_spread,
        "Max_Drawdown": max_drawdown,
        "Max_Drawdown_Abs": max_drawdown_abs,
        "Directional_Accuracy": directional_accuracy,
        "Classification_Accuracy": classification_accuracy,
        "Regime_Dispersion": regime_dispersion,
        "Regime_Stability_Score": regime_dispersion,
        # Derived classification-only stats retained for extensibility.
        "Mean_Rolling_Accuracy": mean_rolling_accuracy,
        "Rolling_Accuracy_Std": rolling_accuracy_std,
    }


def compare_models(model_evaluations: list[dict]) -> pd.DataFrame:
    """Build a ranked model comparison table with composite scoring.

    Args:
        model_evaluations: List of model evaluation payloads.

    Returns:
        Ranked DataFrame sorted by ``Composite_Score`` descending.
    """
    base_columns = [
        "Model",
        "Task_Type",
        "MSE",
        "RMSE",
        "Sharpe",
        "Cumulative_Return",
        "Mean_IC",
        "IC_Std",
        "Directional_Accuracy",
        "Regime_Dispersion",
        "Composite_Score",
        "Rank",
    ]

    if not model_evaluations:
        return pd.DataFrame(columns=base_columns)

    rows = [_extract_model_row(model_eval) for model_eval in model_evaluations]
    comparison_df = pd.DataFrame(rows)

    comparison_df["Composite_Score"] = compute_composite_score(comparison_df)
    has_advanced_components = any(
        col in comparison_df.columns
        and pd.to_numeric(comparison_df[col], errors="coerce").notna().any()
        for col in ["IC_t_stat", "Sharpe_CI_Lower", "Sharpe_CI_Median", "Max_Drawdown_Abs"]
    )
    if has_advanced_components:
        advanced_score = _compute_advanced_composite_score(comparison_df)
        comparison_df["Composite_Score"] = advanced_score.combine_first(
            comparison_df["Composite_Score"]
        )
    comparison_df = comparison_df.sort_values(
        by="Composite_Score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    comparison_df["Rank"] = (
        comparison_df["Composite_Score"]
        .rank(ascending=False, method="dense")
        .astype("Int64")
    )

    for col in base_columns:
        if col not in comparison_df.columns:
            comparison_df[col] = np.nan

    return comparison_df[base_columns]


def summarize_best_model(df: pd.DataFrame) -> dict:
    """Summarize the top-ranked model from a comparison DataFrame."""
    if df.empty or "Composite_Score" not in df.columns:
        return {
            "best_model": "",
            "score": float("nan"),
            "sharpe": float("nan"),
            "mean_ic": float("nan"),
        }

    scored = df.copy()
    scored["Composite_Score"] = pd.to_numeric(scored["Composite_Score"], errors="coerce")
    scored = scored.dropna(subset=["Composite_Score"])
    if scored.empty:
        return {
            "best_model": "",
            "score": float("nan"),
            "sharpe": float("nan"),
            "mean_ic": float("nan"),
        }

    best_row = scored.sort_values("Composite_Score", ascending=False).iloc[0]
    return {
        "best_model": str(best_row.get("Model", "")),
        "score": _to_float(best_row.get("Composite_Score")),
        "sharpe": _to_float(best_row.get("Sharpe")),
        "mean_ic": _to_float(best_row.get("Mean_IC")),
    }
