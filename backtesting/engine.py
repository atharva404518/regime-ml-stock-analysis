from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def run_backtest(
    returns: pd.Series,
    signals: pd.Series,
    holding_period: int = 5,
    position_size: float = 1.0,
    transaction_cost: float = 0.0005,
    threshold: float = 0.0,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Run a non-overlapping long-only backtest from aligned returns and signals.

    Args:
        returns: Realized forward returns aligned to prediction timestamps.
        signals: Model outputs aligned with ``returns`` index.
        holding_period: Number of bars capital is locked after trade entry.
        position_size: Fraction of capital allocated to each trade.
        transaction_cost: Round-trip transaction cost deducted per trade.
        threshold: Minimum signal value required to enter for non-binary signals.
        model_name: Label included in the returned result payload.

    Returns:
        Dictionary containing trade statistics and the strategy equity curve.

    Raises:
        ValueError: If inputs are invalid or indices are not strictly aligned.
        TypeError: If inputs are not pandas Series.
    """
    if not isinstance(returns, pd.Series) or not isinstance(signals, pd.Series):
        raise TypeError("returns and signals must both be pandas Series.")
    if not returns.index.equals(signals.index):
        raise ValueError("Index mismatch: returns and signals must have identical indices.")
    if returns.empty:
        raise ValueError("returns and signals must be non-empty.")
    if holding_period <= 0:
        raise ValueError("holding_period must be a positive integer.")

    aligned_df = pd.concat(
        [returns.rename("returns"), signals.rename("signals")],
        axis=1,
        join="inner",
    )
    if len(aligned_df) != len(returns):
        raise ValueError("Index mismatch detected during alignment.")

    valid_mask = aligned_df["returns"].notna() & aligned_df["signals"].notna()
    returns_values = pd.to_numeric(aligned_df["returns"], errors="coerce").to_numpy(dtype=float)
    signal_values = pd.to_numeric(aligned_df["signals"], errors="coerce").to_numpy(dtype=float)
    valid_values = valid_mask.to_numpy()

    finite_signals = signal_values[np.isfinite(signal_values)]
    if finite_signals.size == 0:
        raise ValueError("signals must contain at least one finite numeric value.")
    is_classification_signal = bool(
        np.nanmin(finite_signals) >= 0.0 and np.nanmax(finite_signals) <= 1.0
    )

    if is_classification_signal:
        raw_position = (signal_values > float(threshold)).astype(float)
    else:
        epsilon = 1e-6
        raw_position = np.where(
            np.abs(signal_values) > epsilon,
            np.sign(signal_values),
            0.0,
        ).astype(float)

    # Shift one bar to avoid lookahead bias.
    shifted_position = (
        pd.Series(raw_position, index=aligned_df.index, dtype=float).shift(1).fillna(0.0)
    )
    shifted_position = shifted_position.where(valid_mask, 0.0)
    position_values = shifted_position.to_numpy(dtype=float)

    LOGGER.info(
        f"Prediction stats | mean={float(np.nanmean(finite_signals)):.6f} std={float(np.nanstd(finite_signals)):.6f}"
    )
    LOGGER.info(
        f"Position distribution: {np.unique(position_values, return_counts=True)}"
    )

    # Enforce non-overlapping trades: if entered at t, skip next holding_period - 1 bars.
    n = len(aligned_df)
    trade_entries = np.zeros(n, dtype=bool)
    trade_positions = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        if valid_values[i] and position_values[i] != 0.0:
            trade_entries[i] = True
            trade_positions[i] = position_values[i]
            i += int(holding_period)
        else:
            i += 1

    strategy_returns = np.zeros(n, dtype=float)
    strategy_returns[trade_entries] = (
        float(position_size) * trade_positions[trade_entries] * returns_values[trade_entries]
        - float(transaction_cost)
    )

    equity_curve = pd.Series(
        np.cumprod(1.0 + strategy_returns),
        index=aligned_df.index,
        name=f"{model_name}_equity_curve",
    )

    num_trades = int(trade_entries.sum())
    cumulative_return = float(equity_curve.iloc[-1] - 1.0)

    annualization_factor = 252.0 / float(n)
    if 1.0 + cumulative_return > 0:
        annualized_return = float((1.0 + cumulative_return) ** annualization_factor - 1.0)
    else:
        annualized_return = float("nan")

    annualized_volatility = float(np.std(strategy_returns, ddof=0) * np.sqrt(252.0))
    sharpe = (
        float(annualized_return / annualized_volatility)
        if annualized_volatility > 0 and np.isfinite(annualized_return)
        else 0.0
    )

    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1.0
    max_drawdown = float(drawdown.min())

    trade_returns = strategy_returns[trade_entries]
    win_rate = float((trade_returns > 0.0).mean()) if num_trades > 0 else 0.0

    return {
        "model_name": model_name,
        "num_trades": num_trades,
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "equity_curve": equity_curve,
    }
