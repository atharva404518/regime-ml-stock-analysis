"""Scoring utilities for cross-model comparison in quantitative research."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_metric(values: list[float], higher_is_better: bool) -> list[float]:
    """Normalize a metric list with min-max scaling and NaN-safe behavior.

    Args:
        values: Raw metric values.
        higher_is_better: Direction of desirability for the metric.

    Returns:
        A list of normalized values with the same length and order as input.
        NaNs are preserved in-place.
    """
    arr = np.asarray(values, dtype=float)
    normalized = np.full(arr.shape, np.nan, dtype=float)

    valid_mask = ~np.isnan(arr)
    if not valid_mask.any():
        return normalized.tolist()

    valid_values = arr[valid_mask]
    min_val = float(np.min(valid_values))
    max_val = float(np.max(valid_values))

    if np.isclose(max_val, min_val):
        normalized[valid_mask] = 0.5
        return normalized.tolist()

    scaled = (valid_values - min_val) / (max_val - min_val)
    if not higher_is_better:
        scaled = 1.0 - scaled

    normalized[valid_mask] = scaled
    return normalized.tolist()


def _normalized_series(
    df: pd.DataFrame,
    column: str,
    higher_is_better: bool,
) -> pd.Series | None:
    """Create a normalized metric series from a DataFrame column when available."""
    if column not in df.columns:
        return None
    numeric = pd.to_numeric(df[column], errors="coerce")
    if not numeric.notna().any():
        return None
    return pd.Series(
        normalize_metric(numeric.tolist(), higher_is_better=higher_is_better),
        index=df.index,
        dtype=float,
    )


def compute_composite_score(df: pd.DataFrame) -> pd.Series:
    """Compute adaptive weighted composite scores from available model metrics.

    The score uses normalized metric components and rebalances weights when a
    component is unavailable across all models.
    """
    normalized_mean_ic = _normalized_series(df, "Mean_IC", higher_is_better=True)
    normalized_sharpe = _normalized_series(df, "Sharpe", higher_is_better=True)
    normalized_cumret = _normalized_series(df, "Cumulative_Return", higher_is_better=True)
    normalized_mse = _normalized_series(df, "MSE", higher_is_better=False)
    normalized_rmse = _normalized_series(df, "RMSE", higher_is_better=False)
    normalized_dir_acc = _normalized_series(df, "Directional_Accuracy", higher_is_better=True)
    normalized_cls_acc = _normalized_series(df, "Classification_Accuracy", higher_is_better=True)

    ic_snr_series = None
    if "Mean_IC" in df.columns and "IC_Std" in df.columns:
        mean_ic = pd.to_numeric(df["Mean_IC"], errors="coerce")
        ic_std = pd.to_numeric(df["IC_Std"], errors="coerce")
        ic_snr_raw = mean_ic / ic_std.where(ic_std > 0, np.nan)
        if ic_snr_raw.notna().any():
            ic_snr_series = pd.Series(
                normalize_metric(ic_snr_raw.tolist(), higher_is_better=True),
                index=df.index,
                dtype=float,
            )

    directional_component = pd.Series(np.nan, index=df.index, dtype=float)
    if normalized_dir_acc is not None:
        directional_component = normalized_dir_acc.copy()
    if normalized_cls_acc is not None:
        if "Task_Type" in df.columns:
            task_type = df["Task_Type"].astype(str).str.lower()
            cls_mask = task_type.eq("classification")
            directional_component = directional_component.where(~cls_mask, normalized_cls_acc)
        directional_component = directional_component.fillna(normalized_cls_acc)

    components: dict[str, tuple[pd.Series, float]] = {}
    if normalized_mean_ic is not None:
        components["Mean_IC"] = (normalized_mean_ic, 0.25)
    if normalized_sharpe is not None:
        components["Sharpe"] = (normalized_sharpe, 0.25)
    if directional_component.notna().any():
        components["Directional_Accuracy"] = (directional_component, 0.15)
    if normalized_cumret is not None:
        components["Cumulative_Return"] = (normalized_cumret, 0.10)

    error_norm_candidates = []
    if normalized_mse is not None:
        error_norm_candidates.append(normalized_mse)
    if normalized_rmse is not None:
        error_norm_candidates.append(normalized_rmse)
    if error_norm_candidates:
        components["Error_Metric"] = (
            pd.concat(error_norm_candidates, axis=1).mean(axis=1, skipna=True),
            0.10,
        )

    if ic_snr_series is not None:
        components["IC_SNR"] = (ic_snr_series, 0.05)

    if not components:
        return pd.Series(np.nan, index=df.index, dtype=float)

    total_component_weight = sum(weight for _series, weight in components.values())
    rebased_weights = {
        name: weight / total_component_weight for name, (_series, weight) in components.items()
    }

    component_df = pd.DataFrame(
        {name: series for name, (series, _weight) in components.items()},
        index=df.index,
    )
    weight_series = pd.Series(rebased_weights, dtype=float)

    weighted_sum = component_df.mul(weight_series, axis=1).sum(axis=1, skipna=True)
    present_weight_sum = component_df.notna().mul(weight_series, axis=1).sum(axis=1)

    composite = weighted_sum / present_weight_sum.where(present_weight_sum > 0, np.nan)

    # Coverage penalty: penalize rows that have sparse metric coverage.
    coverage_ratio = present_weight_sum
    composite = composite * coverage_ratio

    # Multiplicative regime instability penalty (higher dispersion => larger penalty).
    stability_col = (
        "Regime_Stability_Score"
        if "Regime_Stability_Score" in df.columns
        else "Regime_Dispersion"
    )
    regime_dispersion_norm = _normalized_series(df, stability_col, higher_is_better=True)
    if regime_dispersion_norm is not None:
        stability_penalty = (1.0 - regime_dispersion_norm).clip(lower=0.0, upper=1.0)
        composite = composite * stability_penalty.fillna(1.0)

    return composite.astype(float)
