"""Advanced, research-grade model evaluation utilities for quant workflows."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

ANNUALIZATION_FACTOR = np.sqrt(252.0)


def _ensure_series(name: str, value: pd.Series) -> None:
    """Validate that value is a pandas Series."""
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")


def _align_series(
    left: pd.Series,
    right: pd.Series,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    """Align two series on index and drop rows with NaNs."""
    _ensure_series(left_name, left)
    _ensure_series(right_name, right)
    if not left.index.equals(right.index):
        raise ValueError(f"Index mismatch between {left_name} and {right_name}.")

    aligned = pd.concat(
        [pd.to_numeric(left, errors="coerce").rename(left_name),
         pd.to_numeric(right, errors="coerce").rename(right_name)],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError(f"No valid aligned observations for {left_name} and {right_name}.")
    return aligned


def _safe_sharpe(series: pd.Series) -> float:
    """Compute annualized Sharpe ratio with safe zero-division handling."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    std = float(values.std(ddof=0))
    if std == 0.0:
        return float("nan")
    return float((values.mean() / std) * ANNUALIZATION_FACTOR)


def compute_ic_statistics(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 60,
) -> dict[str, Any]:
    """Compute information coefficient (IC) summary statistics.

    Args:
        y_true: True forward returns.
        y_pred: Model predictions aligned with y_true.
        window: Rolling window used for Pearson IC estimation.

    Returns:
        Dictionary containing mean/std/t-stat/hit ratio and rolling IC series.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1.")

    aligned = _align_series(y_true, y_pred, "y_true", "y_pred")
    rolling_ic = aligned["y_true"].rolling(window=window, min_periods=window).corr(
        aligned["y_pred"]
    )
    rolling_ic = rolling_ic.replace([np.inf, -np.inf], np.nan).dropna()

    n = int(len(rolling_ic))
    if n == 0:
        return {
            "mean_ic": float("nan"),
            "ic_std": float("nan"),
            "ic_t_stat": float("nan"),
            "ic_p_value": float("nan"),
            "ic_hit_ratio": float("nan"),
            "rolling_ic": rolling_ic,
        }

    mean_ic = float(rolling_ic.mean())
    ic_std = float(rolling_ic.std(ddof=0))
    if ic_std == 0.0 or n <= 1:
        ic_t_stat = float("nan")
    else:
        ic_t_stat = float(mean_ic / (ic_std / np.sqrt(n)))
    ic_p_value = (
        float(math.erfc(abs(ic_t_stat) / np.sqrt(2.0)))
        if np.isfinite(ic_t_stat)
        else float("nan")
    )

    ic_hit_ratio = float((rolling_ic > 0).mean())

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_t_stat": ic_t_stat,
        "ic_p_value": ic_p_value,
        "ic_hit_ratio": ic_hit_ratio,
        "rolling_ic": rolling_ic,
    }


def compute_ic_decay(
    y_true: pd.Series,
    y_pred: pd.Series,
    lags: tuple[int, ...] = (0, 1, 5),
) -> dict[str, float]:
    """Compute IC decay curve by correlating predictions with lagged future targets."""
    aligned = _align_series(y_true, y_pred, "y_true", "y_pred")
    decay: dict[str, float] = {}
    for lag in lags:
        if lag < 0:
            raise ValueError("IC decay lag values must be non-negative.")
        shifted_target = aligned["y_true"].shift(-lag)
        lag_df = pd.concat([aligned["y_pred"], shifted_target.rename("y_true_shifted")], axis=1).dropna()
        if lag_df.empty:
            decay[f"ic_t_plus_{lag}"] = float("nan")
            continue
        decay[f"ic_t_plus_{lag}"] = float(lag_df["y_pred"].corr(lag_df["y_true_shifted"]))
    return decay


def compute_quantile_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Compute quantile-based return diagnostics by prediction rank.

    Args:
        y_true: True forward returns.
        y_pred: Prediction scores aligned with y_true.
        n_bins: Number of quantile bins.

    Returns:
        DataFrame indexed by quantile bucket with return and Sharpe diagnostics.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    aligned = _align_series(y_true, y_pred, "y_true", "y_pred")
    quantiles = pd.qcut(aligned["y_pred"], q=n_bins, duplicates="drop")
    if quantiles.isna().all():
        raise ValueError("Unable to form quantiles from predictions.")

    eval_df = aligned.copy()
    eval_df["quantile"] = quantiles

    grouped = eval_df.groupby("quantile", observed=False)["y_true"]
    output = grouped.agg(mean_return="mean", std_return="std", count="size")
    output["std_return"] = pd.to_numeric(output["std_return"], errors="coerce").fillna(0.0)
    output["sharpe"] = np.where(
        output["std_return"] > 0,
        (output["mean_return"] / output["std_return"]) * ANNUALIZATION_FACTOR,
        np.nan,
    )
    output = output.replace([np.inf, -np.inf], np.nan)
    return output


def compute_quantile_spread(quantile_df: pd.DataFrame) -> float:
    """Compute top-minus-bottom quantile mean-return spread (Q5 - Q1)."""
    if not isinstance(quantile_df, pd.DataFrame) or quantile_df.empty:
        return float("nan")
    if "mean_return" not in quantile_df.columns:
        return float("nan")
    mean_ret = pd.to_numeric(quantile_df["mean_return"], errors="coerce").dropna()
    if mean_ret.size < 2:
        return float("nan")
    return float(mean_ret.iloc[-1] - mean_ret.iloc[0])


def compute_signal_diagnostics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute signal quality diagnostics for predictions vs target."""
    aligned = _align_series(y_true, y_pred, "y_true", "y_pred")
    pred_std = float(aligned["y_pred"].std(ddof=0))
    true_std = float(aligned["y_true"].std(ddof=0))
    snr = float(pred_std / true_std) if true_std != 0.0 else float("nan")

    pearson_corr = float(aligned["y_true"].corr(aligned["y_pred"], method="pearson"))
    spearman_corr = float(aligned["y_true"].corr(aligned["y_pred"], method="spearman"))

    return {
        "prediction_std": pred_std,
        "true_std": true_std,
        "signal_to_noise_ratio": snr,
        "correlation": pearson_corr,
        "spearman_rank_correlation": spearman_corr,
    }


def compute_trading_diagnostics(
    returns: pd.Series,
    signals: pd.Series,
) -> dict[str, float]:
    """Compute trading diagnostics from returns and model signals.

    Strategy returns are defined as: sign(signals) * returns.
    """
    aligned = _align_series(returns, signals, "returns", "signals")
    positions = np.sign(aligned["signals"]).astype(float)
    strategy_returns = positions * aligned["returns"]
    strategy_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    position_changes = positions.diff().fillna(0.0).ne(0.0)
    turnover = (
        float(position_changes.iloc[1:].mean())
        if len(position_changes) > 1
        else 0.0
    )

    active_mask = positions.ne(0.0)
    active_returns = strategy_returns[active_mask]
    if active_returns.empty:
        win_rate = 0.0
        avg_trade_return = 0.0
        profit_factor = 0.0
    else:
        win_rate = float((active_returns > 0).mean())
        avg_trade_return = float(active_returns.mean())
        gross_profit = float(active_returns[active_returns > 0].sum())
        gross_loss = float(-active_returns[active_returns < 0].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("nan")

    equity_curve = (1.0 + strategy_returns).cumprod()
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1.0
    max_drawdown = float(drawdown.min())
    annualized_sharpe = _safe_sharpe(strategy_returns)

    return {
        "turnover": turnover,
        "win_rate": win_rate,
        "average_trade_return": avg_trade_return,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "annualized_sharpe": annualized_sharpe,
    }


def compute_stability_metrics(
    returns: pd.Series,
    window: int = 60,
) -> dict[str, Any]:
    """Compute rolling stability diagnostics for a return series."""
    _ensure_series("returns", returns)
    if window <= 1:
        raise ValueError("window must be greater than 1.")

    ret = pd.to_numeric(returns, errors="coerce").dropna()
    if ret.empty:
        raise ValueError("returns must contain at least one valid numeric value.")
    if window > len(ret):
        raise ValueError("window cannot exceed number of valid return observations.")

    rolling_mean = ret.rolling(window=window, min_periods=window).mean()
    rolling_std = ret.rolling(window=window, min_periods=window).std(ddof=0)
    rolling_sharpe = (rolling_mean / rolling_std.replace(0.0, np.nan)) * ANNUALIZATION_FACTOR
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

    equity_curve = (1.0 + ret).cumprod()
    rolling_peak = equity_curve.rolling(window=window, min_periods=window).max()
    rolling_drawdown = equity_curve / rolling_peak - 1.0
    rolling_drawdown = rolling_drawdown.replace([np.inf, -np.inf], np.nan)

    sharpe_std = float(rolling_sharpe.dropna().std(ddof=0))
    max_rolling_drawdown = float(rolling_drawdown.dropna().min())

    return {
        "rolling_sharpe": rolling_sharpe,
        "rolling_drawdown": rolling_drawdown,
        "sharpe_std": sharpe_std,
        "max_rolling_drawdown": max_rolling_drawdown,
    }


def bootstrap_sharpe_confidence_interval(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Estimate Sharpe confidence interval via bootstrap resampling."""
    _ensure_series("returns", returns)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1.")

    sample = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if sample.size == 0:
        raise ValueError("returns must contain at least one valid numeric value.")

    rng = np.random.default_rng(42)
    sharpe_samples = []
    for _ in range(n_bootstrap):
        boot = rng.choice(sample, size=sample.size, replace=True)
        boot_std = float(np.std(boot, ddof=0))
        if boot_std == 0.0:
            sharpe_samples.append(np.nan)
        else:
            sharpe_samples.append(float((np.mean(boot) / boot_std) * ANNUALIZATION_FACTOR))

    sharpe_arr = np.asarray(sharpe_samples, dtype=float)
    sharpe_arr = sharpe_arr[np.isfinite(sharpe_arr)]
    if sharpe_arr.size == 0:
        return {
            "sharpe_lower": float("nan"),
            "sharpe_median": float("nan"),
            "sharpe_upper": float("nan"),
        }

    alpha = 1.0 - confidence
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    return {
        "sharpe_lower": float(np.percentile(sharpe_arr, lower_q)),
        "sharpe_median": float(np.percentile(sharpe_arr, 50.0)),
        "sharpe_upper": float(np.percentile(sharpe_arr, upper_q)),
    }


def evaluate_model_advanced(
    y_true: pd.Series,
    y_pred: pd.Series,
    strategy_returns: pd.Series,
) -> dict[str, Any]:
    """Run full advanced evaluation suite for a model.

    Args:
        y_true: True forward returns.
        y_pred: Model predictions aligned to y_true.
        strategy_returns: Realized strategy returns aligned to y_true/y_pred index.

    Returns:
        Structured dictionary containing all advanced diagnostics.
    """
    _ensure_series("strategy_returns", strategy_returns)
    if not y_true.index.equals(y_pred.index):
        raise ValueError("Index mismatch between y_true and y_pred.")
    if not y_true.index.equals(strategy_returns.index):
        raise ValueError("Index mismatch: strategy_returns must align with y_true/y_pred.")

    ic_stats = compute_ic_statistics(y_true=y_true, y_pred=y_pred, window=60)
    quantile_performance = compute_quantile_performance(y_true=y_true, y_pred=y_pred, n_bins=5)
    ic_decay = compute_ic_decay(y_true=y_true, y_pred=y_pred, lags=(0, 1, 5))
    quantile_spread = compute_quantile_spread(quantile_performance)
    signal_diagnostics = compute_signal_diagnostics(y_true=y_true, y_pred=y_pred)
    trading_diagnostics = compute_trading_diagnostics(
        returns=strategy_returns,
        signals=y_pred,
    )
    stability_metrics = compute_stability_metrics(returns=strategy_returns, window=60)
    bootstrap_sharpe = bootstrap_sharpe_confidence_interval(
        returns=strategy_returns,
        n_bootstrap=1000,
        confidence=0.95,
    )

    return {
        "ic_stats": ic_stats,
        "quantile_performance": quantile_performance,
        "quantile_spread": quantile_spread,
        "ic_decay": ic_decay,
        "signal_diagnostics": signal_diagnostics,
        "trading_diagnostics": trading_diagnostics,
        "stability_metrics": stability_metrics,
        "bootstrap_sharpe": bootstrap_sharpe,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_obs = 600
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")

    y_true_synth = pd.Series(rng.normal(0.0005, 0.02, n_obs), index=idx)
    y_pred_synth = pd.Series(
        0.2 * y_true_synth.to_numpy() + rng.normal(0.0, 0.015, n_obs),
        index=idx,
    )
    strategy_returns_synth = pd.Series(
        np.sign(y_pred_synth.to_numpy()) * y_true_synth.to_numpy(),
        index=idx,
    )

    _ = evaluate_model_advanced(
        y_true=y_true_synth,
        y_pred=y_pred_synth,
        strategy_returns=strategy_returns_synth,
    )
