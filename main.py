"""CLI orchestrator for model training, evaluation, comparison, and backtesting."""

from __future__ import annotations

import argparse
import logging
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from config import setup_logging

from backtesting.engine import run_backtest
from evaluation.advanced_evaluation import evaluate_model_advanced
from evaluation.model_summary import compare_models, summarize_best_model
from evaluation.regime_metrics import evaluate_by_regime
from evaluation.rolling_metrics import compute_rolling_metrics
from evaluation.scoring import compute_composite_score

ALLOWED_MODELS = ["LinearRegression", "Ridge", "RandomForest", "LSTM"]
ALLOWED_MODEL_KEYS = ["linear", "ridge", "random_forest", "lstm"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the quant evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Quantitative research CLI: train, evaluate, backtest, and compare models."
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol (example: AAPL).")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "linear", "ridge", "random_forest", "lstm"],
        help="Model to run. Default runs all models.",
    )
    parser.add_argument(
        "--holding_period",
        type=int,
        default=5,
        help="Non-overlapping holding period used by the backtesting engine.",
    )
    parser.add_argument(
        "--transaction_cost",
        type=float,
        default=0.0005,
        help="Round-trip transaction cost per trade.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum signal threshold required to enter long trades.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load data from local cache only (skip API calls).",
    )
    parser.add_argument(
        "--refresh_data",
        action="store_true",
        help="Refresh cached data by prioritizing API ingestion in auto mode.",
    )
    parser.add_argument(
        "--data_mode",
        choices=["auto", "tiingo", "local", "synthetic", "api", "cache"],
        default="auto",
        help="Data source mode. Auto uses hierarchical fallback chain.",
    )
    parser.add_argument(
        "--save_dataset",
        action="store_true",
        help="Save ingested dataset into versioned datasets store.",
    )
    parser.add_argument(
        "--quality_report",
        action="store_true",
        help="Print data quality report before model pipeline execution.",
    )
    parser.add_argument(
        "--use_walk_forward",
        action="store_true",
        help="Enable expanding-window walk-forward validation.",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save experiment results, leaderboard row, and run metadata.",
    )
    parser.add_argument(
        "--no_synthetic",
        action="store_true",
        help="Abort run when synthetic data is detected.",
    )
    parser.add_argument(
        "--analysis_mode",
        choices=["auto", "custom"],
        default="auto",
        help="Auto mode generates comparison plots; custom mode skips plotting.",
    )
    parser.add_argument(
        "--advanced_analysis",
        action="store_true",
        help="Enable advanced evaluation diagnostics output.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run only Ridge and Random Forest.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plotting even when analysis_mode is auto.",
    )
    parser.add_argument(
        "--exclude_model",
        nargs="*",
        default=[],
        choices=["linear", "ridge", "random_forest", "lstm"],
        help="Optional model list to exclude from execution.",
    )
    return parser.parse_args()


def _selected_models(model_arg: str) -> list[str]:
    """Resolve selected model list from CLI argument."""
    if model_arg == "all":
        return ALLOWED_MODEL_KEYS.copy()
    return [model_arg]


def _validate_index_alignment(reference_index: pd.Index, series: pd.Series, name: str) -> None:
    """Fail loudly when index alignment constraints are violated."""
    if not reference_index.equals(series.index):
        raise ValueError(f"Index mismatch detected for {name}.")


def _regression_overall_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute core regression metrics from aligned series."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    directional_accuracy = float((np.sign(y_true) == np.sign(y_pred)).mean())
    return {
        "mse": float(mse),
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
    }


def _rolling_window_length(series_length: int) -> int:
    """Choose a valid rolling window size for current test-set length."""
    if series_length < 2:
        raise ValueError("Test set is too short for rolling analysis (need at least 2 samples).")
    return min(60, series_length)


def _build_model_evaluation(
    model_name: str,
    task_type: str,
    overall_metrics: dict[str, Any],
    y_true: pd.Series,
    y_pred: pd.Series,
    test_df: pd.DataFrame,
    holding_period: int,
    transaction_cost: float,
    threshold: float,
) -> dict[str, Any]:
    """Create a full model evaluation payload using existing evaluation/backtest modules."""
    _validate_index_alignment(y_true.index, y_pred, f"{model_name} predictions")

    regime_df = evaluate_by_regime(
        df=test_df.loc[y_true.index],
        y_true=y_true,
        y_pred=y_pred,
        model_name=model_name,
    )

    rolling_df = compute_rolling_metrics(
        y_true=y_true,
        y_pred=y_pred,
        window=_rolling_window_length(len(y_true)),
        task_type=task_type,
        model_name=model_name,
    )

    trading_metrics = run_backtest(
        returns=test_df.loc[y_true.index, "target_5d"],
        signals=y_pred,
        holding_period=holding_period,
        transaction_cost=transaction_cost,
        threshold=threshold,
        model_name=model_name,
    )
    equity_curve = trading_metrics.get("equity_curve")
    if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
        raise ValueError(f"Missing or empty equity_curve for {model_name} backtest.")
    strategy_returns = equity_curve.pct_change().fillna(0.0)
    _validate_index_alignment(y_true.index, strategy_returns, f"{model_name} strategy_returns")

    try:
        advanced_eval = evaluate_model_advanced(
            y_true=y_true,
            y_pred=y_pred,
            strategy_returns=strategy_returns,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Advanced evaluation skipped for %s due to: %s",
            model_name,
            exc,
        )
        advanced_eval = {
            "ic_stats": {
                "mean_ic": float("nan"),
                "ic_std": float("nan"),
                "ic_t_stat": float("nan"),
                "ic_hit_ratio": float("nan"),
                "rolling_ic": pd.Series(dtype=float),
            },
            "quantile_performance": pd.DataFrame(),
            "signal_diagnostics": {},
            "trading_diagnostics": {},
            "stability_metrics": {},
            "bootstrap_sharpe": {},
        }
    if task_type == "classification":
        advanced_eval["ic_stats"] = {
            "mean_ic": float("nan"),
            "ic_std": float("nan"),
            "ic_t_stat": float("nan"),
            "ic_hit_ratio": float("nan"),
            "rolling_ic": pd.Series(dtype=float),
        }

    return {
        "model_name": model_name,
        "task_type": task_type,
        "overall_metrics": overall_metrics,
        "rolling_metrics": rolling_df,
        "regime_metrics": regime_df,
        "trading_metrics": trading_metrics,
        "advanced_evaluation": advanced_eval,
        "y_true": y_true.copy(),
        "y_pred": y_pred.copy(),
    }


def _print_comparison(comparison_df: pd.DataFrame, best_summary: dict[str, Any]) -> None:
    """Print model comparison and best-model summary."""
    print("\nModel Comparison Summary")
    if comparison_df.empty:
        print("No model evaluations available.")
        return
    print(comparison_df.to_string(index=False))

    print("\nBest Model")
    print(f"Model: {best_summary.get('best_model', '')}")
    print(f"Composite Score: {best_summary.get('score', np.nan):.6f}")
    print(f"Sharpe: {best_summary.get('sharpe', np.nan):.6f}")
    print(f"Mean IC: {best_summary.get('mean_ic', np.nan):.6f}")


def _print_advanced_diagnostics(model_evaluations: list[dict[str, Any]]) -> None:
    """Print compact advanced diagnostics per model."""
    print("\nAdvanced Diagnostics")
    for model_eval in model_evaluations:
        advanced = model_eval.get("advanced_evaluation", {})
        ic_stats = advanced.get("ic_stats", {})
        signal_diag = advanced.get("signal_diagnostics", {})
        trading_diag = advanced.get("trading_diagnostics", {})
        bootstrap = advanced.get("bootstrap_sharpe", {})

        print(f"\nModel: {model_eval.get('model_name', 'Unknown')}")
        print(f"IC t-stat: {float(ic_stats.get('ic_t_stat', np.nan)):.6f}")
        print(
            "Signal-to-noise ratio: "
            f"{float(signal_diag.get('signal_to_noise_ratio', np.nan)):.6f}"
        )
        print(f"Profit factor: {float(trading_diag.get('profit_factor', np.nan)):.6f}")
        print(f"Turnover: {float(trading_diag.get('turnover', np.nan)):.6f}")
        print(
            "Sharpe CI lower/upper: "
            f"{float(bootstrap.get('sharpe_lower', np.nan)):.6f} / "
            f"{float(bootstrap.get('sharpe_upper', np.nan)):.6f}"
        )


def _compute_random_baseline(
    test_df: pd.DataFrame | None,
) -> tuple[pd.Series | None, pd.Series | None, float]:
    """Compute random strategy equity and Sharpe baseline from test returns."""
    if not isinstance(test_df, pd.DataFrame) or "target_5d" not in test_df.columns or test_df.empty:
        return None, None, float("nan")

    base_returns = pd.to_numeric(test_df["target_5d"], errors="coerce").dropna()
    if base_returns.empty:
        return None, None, float("nan")

    np.random.seed(42)
    random_signal = np.random.choice([-1, 1], size=len(base_returns))
    random_returns = pd.Series(random_signal, index=base_returns.index, dtype=float) * base_returns
    random_equity = (1.0 + random_returns).cumprod()

    std = float(random_returns.std(ddof=0))
    random_sharpe = float((random_returns.mean() / std) * np.sqrt(252.0)) if std > 0 else float("nan")
    return random_returns, random_equity, random_sharpe


def _normalize_equity_curve(equity_curve: pd.Series) -> pd.Series | None:
    """Normalize a strictly positive equity curve to start at 1.0."""
    series = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if series.empty:
        return None
    first_val = float(series.iloc[0])
    if first_val <= 0:
        return None
    normalized = series / first_val
    normalized = normalized[normalized > 0]
    if normalized.empty:
        return None
    return normalized


def _print_market_efficiency_insight(
    model_evaluations: list[dict[str, Any]],
    best_model_name: str,
    random_sharpe: float,
) -> None:
    """Print simple market-efficiency insight from model vs random Sharpe comparison."""
    best_model_sharpe = float("nan")
    for model_eval in model_evaluations:
        if str(model_eval.get("model_name", "")) == best_model_name:
            best_model_sharpe = float(model_eval.get("trading_metrics", {}).get("sharpe", np.nan))
            break

    print("\nMarket Efficiency Insight")
    print(f"Best Model ({best_model_name}) Sharpe: {best_model_sharpe:.6f}")
    print(f"Random Strategy Sharpe: {random_sharpe:.6f}")

    if np.isfinite(best_model_sharpe) and np.isfinite(random_sharpe):
        diff = best_model_sharpe - random_sharpe
        if diff > 0.10:
            print("Conclusion: Model performance is materially above random, indicating exploitable structure.")
        elif diff > 0.0:
            print("Conclusion: Model is marginally above random; edge may be weak or regime-dependent.")
        else:
            print("Conclusion: Performance is not above random; evidence supports a more efficient market.")
    else:
        print("Conclusion: Insufficient valid Sharpe values for a robust efficiency inference.")


def _plot_performance_dashboard(
    model_evaluations: list[dict[str, Any]],
    ticker: str,
    best_model_name: str,
    buy_hold_equity: pd.Series | None,
    random_equity: pd.Series | None,
) -> None:
    """Plot trading-performance dashboard in a dedicated 2x2 figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax_equity, ax_benchmark, ax_sharpe, ax_drawdown = axes.flatten()

    model_curves: dict[str, pd.Series] = {}
    model_sharpes: dict[str, float] = {}
    for model_eval in model_evaluations:
        model_name = str(model_eval.get("model_name", "Model"))
        equity_curve = model_eval.get("trading_metrics", {}).get("equity_curve")
        if not isinstance(equity_curve, pd.Series) or equity_curve.empty:
            continue
        norm_curve = _normalize_equity_curve(equity_curve)
        if norm_curve is not None and not norm_curve.empty:
            model_curves[model_name] = norm_curve
            model_sharpes[model_name] = float(
                model_eval.get("trading_metrics", {}).get("sharpe", np.nan)
            )

    ranked_models = sorted(
        model_curves.keys(),
        key=lambda name: model_sharpes.get(name, float("-inf")),
        reverse=True,
    )
    top_two_models = ranked_models[:2]
    selected_model_curves = {k: v for k, v in model_curves.items() if k in top_two_models}

    buy_hold_norm = (
        _normalize_equity_curve(buy_hold_equity)
        if isinstance(buy_hold_equity, pd.Series)
        else None
    )
    random_norm = (
        _normalize_equity_curve(random_equity)
        if isinstance(random_equity, pd.Series)
        else None
    )

    global_equity_values: list[float] = []
    for eq in selected_model_curves.values():
        global_equity_values.extend(eq.to_numpy(dtype=float).tolist())
    if isinstance(buy_hold_norm, pd.Series):
        global_equity_values.extend(buy_hold_norm.to_numpy(dtype=float).tolist())
    if isinstance(random_norm, pd.Series):
        global_equity_values.extend(random_norm.to_numpy(dtype=float).tolist())

    # Panel 1: Equity Curves (all models + baselines)
    if selected_model_curves or isinstance(buy_hold_norm, pd.Series) or isinstance(random_norm, pd.Series):
        for model_name, eq in selected_model_curves.items():
            lw = 3.0 if model_name == best_model_name else (1.1 if model_name == "RandomForest" else 1.6)
            alpha = 1.0 if model_name == best_model_name else (0.50 if model_name == "RandomForest" else 0.80)
            ax_equity.plot(eq.index, eq.values, label=model_name, linewidth=lw, alpha=alpha)
        if isinstance(buy_hold_norm, pd.Series):
            ax_equity.plot(
                buy_hold_norm.index,
                buy_hold_norm.values,
                label="Buy & Hold",
                linestyle="--",
                linewidth=2.4,
                color="black",
            )
        if isinstance(random_norm, pd.Series):
            ax_equity.plot(
                random_norm.index,
                random_norm.values,
                label="Random Strategy",
                linestyle=":",
                linewidth=2.0,
                color="gray",
            )
        ax_equity.set_yscale("log")
        ax_equity.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.9)
    else:
        ax_equity.text(0.5, 0.5, "No equity data available", ha="center", va="center")
    ax_equity.set_title(f"{ticker} - Equity Curves (Normalized, Log Scale)", fontsize=12)
    ax_equity.grid(True, alpha=0.3)

    # Panel 2: Benchmark (best vs buy&hold vs random)
    benchmark_has_data = False
    if best_model_name in model_curves:
        best_eq = model_curves[best_model_name]
        ax_benchmark.plot(
            best_eq.index,
            best_eq.values,
            label=f"Best ({best_model_name})",
            linewidth=3.2,
            alpha=1.0,
        )
        benchmark_has_data = True
    if isinstance(buy_hold_norm, pd.Series):
        ax_benchmark.plot(
            buy_hold_norm.index,
            buy_hold_norm.values,
            label="Buy & Hold",
            linestyle="--",
            linewidth=2.4,
            color="black",
        )
        benchmark_has_data = True
    if isinstance(random_norm, pd.Series):
        ax_benchmark.plot(
            random_norm.index,
            random_norm.values,
            label="Random Strategy",
            linestyle=":",
            linewidth=2.0,
            color="gray",
        )
        benchmark_has_data = True
    if benchmark_has_data:
        ax_benchmark.set_yscale("log")
        ax_benchmark.legend(loc="upper left", fontsize=9, framealpha=0.9)
    else:
        ax_benchmark.text(0.5, 0.5, "No benchmark data available", ha="center", va="center")
    ax_benchmark.set_title(f"{ticker} - Benchmark Comparison", fontsize=12)
    ax_benchmark.grid(True, alpha=0.3)

    # Keep consistent y-axis scaling between log-equity subplots.
    if global_equity_values:
        positive_vals = np.asarray([v for v in global_equity_values if v > 0], dtype=float)
        if positive_vals.size > 0:
            y_min = float(np.nanmin(positive_vals))
            y_max = float(np.nanmax(positive_vals))
            if y_min > 0 and y_max > y_min:
                ax_equity.set_ylim(y_min, y_max)
                ax_benchmark.set_ylim(y_min, y_max)

    # Panel 3: Rolling Sharpe.
    sharpe_has_data = False
    for model_name, eq in selected_model_curves.items():
        daily_returns = eq.pct_change().fillna(0.0)
        rolling_mean = daily_returns.rolling(window=60, min_periods=60).mean()
        rolling_std = daily_returns.rolling(window=60, min_periods=60).std(ddof=0)
        rolling_sharpe = (
            (rolling_mean / rolling_std.replace(0.0, np.nan)) * np.sqrt(252.0)
        ).dropna()
        if rolling_sharpe.empty:
            continue
        sharpe_ema = rolling_sharpe.ewm(span=20, adjust=False).mean()
        ax_sharpe.plot(
            rolling_sharpe.index,
            rolling_sharpe.values,
            label=f"{model_name} (raw)",
            linewidth=1.0,
            alpha=0.25,
        )
        lw = 2.8 if model_name == best_model_name else 1.8
        alpha = 0.95 if model_name == best_model_name else 0.80
        ax_sharpe.plot(
            sharpe_ema.index,
            sharpe_ema.values,
            label=f"{model_name} (EMA20)",
            linewidth=lw,
            alpha=alpha,
        )
        sharpe_has_data = True
    if sharpe_has_data:
        ax_sharpe.legend(loc="upper left", fontsize=9, framealpha=0.9)
    else:
        ax_sharpe.text(0.5, 0.5, "No rolling Sharpe data", ha="center", va="center")
    ax_sharpe.set_title(f"{ticker} - Rolling Sharpe (60)", fontsize=12)
    ax_sharpe.grid(True, alpha=0.3)

    # Panel 4: Rolling Drawdown.
    drawdown_has_data = False
    for model_name, eq in selected_model_curves.items():
        drawdown = (eq / eq.cummax() - 1.0).dropna()
        if drawdown.empty:
            continue
        lw = 2.5 if model_name == best_model_name else (1.1 if model_name == "RandomForest" else 1.4)
        alpha = 0.95 if model_name == best_model_name else (0.50 if model_name == "RandomForest" else 0.80)
        ax_drawdown.plot(drawdown.index, drawdown.values, label=model_name, linewidth=lw, alpha=alpha)
        min_idx = drawdown.idxmin()
        min_val = float(drawdown.min())
        ax_drawdown.scatter([min_idx], [min_val], s=22, alpha=0.9)
        ax_drawdown.annotate(
            f"{model_name} MDD {min_val:.2%}",
            xy=(min_idx, min_val),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=8,
        )
        drawdown_has_data = True
    if isinstance(buy_hold_norm, pd.Series):
        buy_hold_drawdown = (buy_hold_norm / buy_hold_norm.cummax() - 1.0).dropna()
        if not buy_hold_drawdown.empty:
            ax_drawdown.plot(
                buy_hold_drawdown.index,
                buy_hold_drawdown.values,
                label="Buy & Hold",
                linestyle="--",
                linewidth=2.2,
                color="black",
            )
            drawdown_has_data = True
    if isinstance(random_norm, pd.Series):
        random_drawdown = (random_norm / random_norm.cummax() - 1.0).dropna()
        if not random_drawdown.empty:
            ax_drawdown.plot(
                random_drawdown.index,
                random_drawdown.values,
                label="Random Strategy",
                linestyle=":",
                linewidth=1.9,
                color="gray",
            )
            drawdown_has_data = True
    if drawdown_has_data:
        ax_drawdown.legend(loc="lower left", fontsize=9, framealpha=0.9)
    else:
        ax_drawdown.text(0.5, 0.5, "No rolling drawdown data", ha="center", va="center")
    ax_drawdown.set_title(f"{ticker} - Rolling Drawdown", fontsize=12)
    ax_drawdown.grid(True, alpha=0.3)

    fig.suptitle(f"{ticker} - Trading Performance Dashboard", fontsize=15)
    plt.show()


def _plot_diagnostics_dashboard(
    model_evaluations: list[dict[str, Any]],
    ticker: str,
    best_model_name: str,
    test_df: pd.DataFrame | None = None,
) -> None:
    """Plot model-diagnostics dashboard in a clean 3x2 figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(17, 13), constrained_layout=True)
    (
        ax_ic,
        ax_regime_trend,
        ax_regime_vol,
        ax_quantile,
        ax_regime_stability,
        ax_scatter,
    ) = axes.flatten()

    # Rolling IC (classification IC is excluded by design upstream).
    has_ic = False
    for model_eval in model_evaluations:
        if str(model_eval.get("task_type", "")).lower() != "regression":
            continue
        rolling_ic = model_eval.get("advanced_evaluation", {}).get("ic_stats", {}).get("rolling_ic")
        if not isinstance(rolling_ic, pd.Series) or rolling_ic.empty:
            continue
        ic = pd.to_numeric(rolling_ic, errors="coerce").dropna()
        if ic.empty:
            continue
        model_name = str(model_eval.get("model_name", "Model"))
        lw = 2.3 if model_name == best_model_name else 1.5
        alpha = 0.95 if model_name == best_model_name else 0.75
        ax_ic.plot(ic.index, ic.values, label=model_name, linewidth=lw, alpha=alpha)
        has_ic = True
        mean_ic = float(ic.mean())
        std_ic = float(ic.std(ddof=0))
        ax_ic.axhline(mean_ic, linestyle="--", linewidth=1.2, alpha=0.8, color="black")
        ax_ic.fill_between(
            ic.index,
            mean_ic - std_ic,
            mean_ic + std_ic,
            alpha=0.12,
            color="gray",
        )
    if has_ic:
        ax_ic.legend(loc="upper left", fontsize=9, framealpha=0.9)
    else:
        ax_ic.text(0.5, 0.5, "No rolling IC data", ha="center", va="center")
    ax_ic.set_title(f"{ticker} - Rolling Information Coefficient", fontsize=12)
    ax_ic.grid(True, alpha=0.3)

    # Regime performance (Trend and Volatility).
    regime_frames = []
    for model_eval in model_evaluations:
        regime_df = model_eval.get("regime_metrics")
        if isinstance(regime_df, pd.DataFrame) and not regime_df.empty:
            regime_frames.append(regime_df.copy())
    if regime_frames:
        all_regimes = pd.concat(regime_frames, ignore_index=True)
        trend_regimes = all_regimes[all_regimes["Regime_Type"] == "Trend"].copy()
        trend_pivot = trend_regimes.pivot_table(
            index="Regime_Value",
            columns="Model",
            values="Directional_Accuracy",
            aggfunc="mean",
        )
        ordered_regimes = [reg for reg in ["Bull", "Bear", "Sideways"] if reg in trend_pivot.index]
        if ordered_regimes:
            trend_pivot = trend_pivot.reindex(ordered_regimes)
        if not trend_pivot.empty:
            trend_pivot.plot(kind="bar", ax=ax_regime_trend)
            ax_regime_trend.legend(title="Model", loc="upper left", fontsize=8, framealpha=0.9)
        else:
            ax_regime_trend.text(0.5, 0.5, "No trend regime data", ha="center", va="center")

        vol_regimes = all_regimes[all_regimes["Regime_Type"] == "Volatility"].copy()
        vol_pivot = vol_regimes.pivot_table(
            index="Regime_Value",
            columns="Model",
            values="Directional_Accuracy",
            aggfunc="mean",
        )
        if not vol_pivot.empty:
            vol_pivot.plot(kind="bar", ax=ax_regime_vol)
            ax_regime_vol.legend(title="Model", loc="upper left", fontsize=8, framealpha=0.9)
        else:
            ax_regime_vol.text(0.5, 0.5, "No volatility regime data", ha="center", va="center")
    else:
        ax_regime_trend.text(0.5, 0.5, "No regime metrics", ha="center", va="center")
        ax_regime_vol.text(0.5, 0.5, "No regime metrics", ha="center", va="center")
    ax_regime_trend.set_title(f"{ticker} - Trend Regime Performance", fontsize=12)
    ax_regime_trend.grid(True, alpha=0.3)
    ax_regime_vol.set_title(f"{ticker} - Volatility Regime Performance", fontsize=12)
    ax_regime_vol.grid(True, alpha=0.3)

    # Quantile monotonicity.
    has_quantile = False
    for model_eval in model_evaluations:
        quantile_df = model_eval.get("advanced_evaluation", {}).get("quantile_performance")
        if not isinstance(quantile_df, pd.DataFrame) or quantile_df.empty or "mean_return" not in quantile_df.columns:
            continue
        q = pd.to_numeric(quantile_df["mean_return"], errors="coerce").dropna()
        if q.empty:
            continue
        x = np.arange(1, len(q) + 1)
        model_name = str(model_eval.get("model_name", "Model"))
        lw = 2.3 if model_name == best_model_name else 1.4
        alpha = 0.95 if model_name == best_model_name else 0.75
        ax_quantile.plot(x, q.to_numpy(dtype=float), marker="o", linewidth=lw, alpha=alpha, label=model_name)
        has_quantile = True
    if has_quantile:
        ax_quantile.legend(loc="upper left", fontsize=9, framealpha=0.9)
    else:
        ax_quantile.text(0.5, 0.5, "No quantile data", ha="center", va="center")
    ax_quantile.set_title(f"{ticker} - Quantile Mean Return", fontsize=12)
    ax_quantile.set_xlabel("Quantile (Low -> High Signal)")
    ax_quantile.grid(True, alpha=0.3)

    # Regime stability curve: rolling Sharpe by regime.
    has_regime_stability = False
    best_eval_for_regime = next(
        (m for m in model_evaluations if str(m.get("model_name", "")) == best_model_name),
        None,
    )
    if (
        best_eval_for_regime is not None
        and isinstance(test_df, pd.DataFrame)
        and "Trend_Regime" in test_df.columns
    ):
        eq = best_eval_for_regime.get("trading_metrics", {}).get("equity_curve")
        if isinstance(eq, pd.Series) and not eq.empty:
            returns = pd.to_numeric(eq, errors="coerce").pct_change()
            regime_series = test_df.reindex(returns.index).get("Trend_Regime")
            if regime_series is not None:
                for regime_name in ["Bull", "Bear", "Sideways"]:
                    reg_returns = returns.where(regime_series == regime_name, np.nan)
                    roll_mean = reg_returns.rolling(window=60, min_periods=20).mean()
                    roll_std = reg_returns.rolling(window=60, min_periods=20).std(ddof=0)
                    reg_sharpe = ((roll_mean / roll_std.replace(0.0, np.nan)) * np.sqrt(252.0)).dropna()
                    if reg_sharpe.empty:
                        continue
                    ax_regime_stability.plot(
                        reg_sharpe.index,
                        reg_sharpe.values,
                        label=regime_name,
                        linewidth=1.8,
                    )
                    has_regime_stability = True
    if has_regime_stability:
        ax_regime_stability.legend(loc="upper left", fontsize=9, framealpha=0.9)
    else:
        ax_regime_stability.text(0.5, 0.5, "No regime stability data", ha="center", va="center")
    ax_regime_stability.set_title(f"{ticker} - Regime Stability Curve", fontsize=12)
    ax_regime_stability.grid(True, alpha=0.3)

    # Actual vs Predicted (regression models only).
    regression_best_eval = next(
        (
            m
            for m in model_evaluations
            if str(m.get("model_name", "")) == best_model_name
            and str(m.get("task_type", "")).lower() == "regression"
        ),
        None,
    )
    if regression_best_eval is None:
        regression_best_eval = next(
            (m for m in model_evaluations if str(m.get("task_type", "")).lower() == "regression"),
            None,
        )

    has_scatter = False
    if regression_best_eval is not None:
        scatter_model_name = str(regression_best_eval.get("model_name", "Regression"))
        y_true_series = regression_best_eval.get("y_true")
        y_pred_series = regression_best_eval.get("y_pred")
        if isinstance(y_true_series, pd.Series) and isinstance(y_pred_series, pd.Series):
            aligned = pd.concat(
                [
                    pd.to_numeric(y_true_series, errors="coerce").rename("y_true"),
                    pd.to_numeric(y_pred_series, errors="coerce").rename("y_pred"),
                ],
                axis=1,
            ).dropna()
            if not aligned.empty:
                ax_scatter.scatter(
                    aligned["y_true"],
                    aligned["y_pred"],
                    s=14,
                    alpha=0.5,
                    label=scatter_model_name,
                )
                low = float(min(aligned["y_true"].min(), aligned["y_pred"].min()))
                high = float(max(aligned["y_true"].max(), aligned["y_pred"].max()))
                ax_scatter.plot([low, high], [low, high], "r--", linewidth=1.5, label="Perfect")
                ax_scatter.legend(loc="upper left", fontsize=9, framealpha=0.9)
                has_scatter = True
    if not has_scatter:
        ax_scatter.text(0.5, 0.5, "No regression scatter available", ha="center", va="center")
    ax_scatter.set_title(f"{ticker} - Actual vs Predicted (Regression)", fontsize=12)
    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.grid(True, alpha=0.3)

    fig.suptitle(f"{ticker} - Model Diagnostics Dashboard", fontsize=15)
    plt.show()


def _plot_auto_analysis(
    model_evaluations: list[dict[str, Any]],
    logger: logging.Logger,
    test_df: pd.DataFrame | None = None,
    best_model_name: str | None = None,
    ticker: str = "Ticker",
) -> None:
    """Render two figures: trading performance and model diagnostics."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        logger.warning("matplotlib not installed; skipping auto analysis plots.")
        return

    if not model_evaluations:
        logger.warning("No model evaluations available for plotting.")
        return

    if not best_model_name:
        best_model_name = str(model_evaluations[0].get("model_name", ""))

    _, random_equity, _ = _compute_random_baseline(test_df=test_df)
    buy_hold_equity = None
    if isinstance(test_df, pd.DataFrame) and "target_5d" in test_df.columns and not test_df.empty:
        buy_hold_returns = pd.to_numeric(test_df["target_5d"], errors="coerce").dropna()
        if not buy_hold_returns.empty:
            buy_hold_equity = (1.0 + buy_hold_returns).cumprod()

    _plot_performance_dashboard(
        model_evaluations=model_evaluations,
        ticker=ticker,
        best_model_name=best_model_name,
        buy_hold_equity=buy_hold_equity,
        random_equity=random_equity,
    )
    _plot_diagnostics_dashboard(
        model_evaluations=model_evaluations,
        ticker=ticker,
        best_model_name=best_model_name,
        test_df=test_df,
    )



def main() -> None:
    """Run the full research evaluation + backtesting orchestration pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    if args.holding_period <= 0:
        raise ValueError("--holding_period must be a positive integer.")
    if args.transaction_cost < 0:
        raise ValueError("--transaction_cost must be non-negative.")

    from data.loader import load_market_data_with_report
    from data.dataset_manager import save_versioned_dataset
    from experiments.tracker import append_leaderboard, append_run_log, save_run_results
    from features.feature_engineering import create_features
    from models.lstm_model import predict_lstm, train_lstm_model
    from models.baseline_model import (
        evaluate_regression,
        prepare_xy_regression,
        train_linear_regression,
        train_random_forest_model,
        train_ridge_model,
        train_test_split_time_series,
    )
    from regimes.regime_engine import add_market_regimes

    requested_mode = args.data_mode
    if requested_mode == "api":
        requested_mode = "tiingo"
    if requested_mode == "cache":
        requested_mode = "local"
    if args.offline and requested_mode in {"auto", "tiingo"}:
        logger.info("Offline flag detected; switching data mode to local.")
        requested_mode = "local"
    if args.refresh_data and requested_mode == "auto":
        logger.info("Refresh flag detected; prioritizing Tiingo source.")
        requested_mode = "tiingo"

    data, quality = load_market_data_with_report(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        mode=requested_mode,
    )
    data_source = str(data.attrs.get("source", "unknown")).lower()
    logger.info("Data source: %s", data_source)
    dataset_metadata = data.attrs.get("dataset_metadata", {})
    is_synthetic = bool(dataset_metadata.get("is_synthetic", data_source == "synthetic"))
    if is_synthetic:
        print("EXCLUDE THIS RUN FROM FINAL ANALYSIS")
        if args.no_synthetic:
            raise ValueError("Synthetic data detected with --no_synthetic enabled.")
    if args.quality_report:
        logger.info("Data Quality Report: %s", quality)
    dataset_version = "na"
    if args.save_dataset:
        version_record = save_versioned_dataset(
            df=data,
            ticker=args.ticker.strip().upper(),
            start=args.start,
            end=args.end,
            source=data_source,
            datasets_root=Path(__file__).resolve().parent / "data" / "datasets",
        )
        dataset_version = str(version_record.get("version", "na"))
        logger.info("Saved dataset version: %s", version_record)

    feature_df = create_features(data)
    feature_df = add_market_regimes(feature_df)
    train_df, test_df = train_test_split_time_series(feature_df, test_size=0.2)

    X_train_reg, y_train_reg = prepare_xy_regression(train_df)
    X_test_reg, y_test_reg = prepare_xy_regression(test_df)

    if args.quick:
        selected_models = ["ridge", "random_forest"]
        logger.info("Quick mode enabled: running Ridge and Random Forest only.")
    else:
        selected_models = _selected_models(args.model)
    excluded_models = {model.strip().lower() for model in args.exclude_model if model.strip()}
    if excluded_models:
        selected_models = [model for model in selected_models if model not in excluded_models]
        logger.info("Excluded models: %s", sorted(excluded_models))
    if not selected_models:
        raise ValueError("No models selected to run after applying --model/--quick and --exclude_model.")

    walk_forward_comparison_df = pd.DataFrame()
    walk_forward_fold_df = pd.DataFrame()
    if args.use_walk_forward:
        logger.info("Walk-forward validation enabled (expanding window by year).")
        years = sorted(pd.to_datetime(feature_df.index).year.unique().tolist())
        fold_rows: list[dict[str, Any]] = []
        min_train_years = 4
        if len(years) <= min_train_years:
            logger.warning("Insufficient yearly history for walk-forward validation.")
        else:
            for idx in range(min_train_years, len(years)):
                train_years = years[:idx]
                test_year = years[idx]
                fold_train = feature_df[pd.to_datetime(feature_df.index).year.isin(train_years)].copy()
                fold_test = feature_df[pd.to_datetime(feature_df.index).year == test_year].copy()
                if len(fold_train) < 120 or len(fold_test) < 30:
                    continue

                fold_label = f"{train_years[0]}-{train_years[-1]}->{test_year}"
                X_train_reg_wf, y_train_reg_wf = prepare_xy_regression(fold_train)
                X_test_reg_wf, y_test_reg_wf = prepare_xy_regression(fold_test)

                for model_key in selected_models:
                    try:
                        model_name = ""
                        task_type = "regression"
                        if model_key == "linear":
                            model_name = "LinearRegression"
                            wf_model = train_linear_regression(X_train_reg_wf, y_train_reg_wf)
                            preds = pd.Series(wf_model.predict(X_test_reg_wf), index=X_test_reg_wf.index)
                            y_true = y_test_reg_wf
                        elif model_key == "ridge":
                            model_name = "Ridge"
                            _a, _r, wf_model, wf_scaler = train_ridge_model(
                                X_train_reg_wf, y_train_reg_wf, X_test_reg_wf, y_test_reg_wf
                            )
                            preds = pd.Series(
                                wf_model.predict(wf_scaler.transform(X_test_reg_wf)),
                                index=X_test_reg_wf.index,
                            )
                            y_true = y_test_reg_wf
                        elif model_key == "random_forest":
                            model_name = "RandomForest"
                            _m, wf_model = train_random_forest_model(
                                X_train_reg_wf, y_train_reg_wf, X_test_reg_wf, y_test_reg_wf
                            )
                            preds = pd.Series(wf_model.predict(X_test_reg_wf), index=X_test_reg_wf.index)
                            y_true = y_test_reg_wf
                        elif model_key == "lstm":
                            model_name = "LSTM"
                            wf_scaler = StandardScaler()
                            X_train_scaled = pd.DataFrame(
                                wf_scaler.fit_transform(X_train_reg_wf),
                                index=X_train_reg_wf.index,
                                columns=X_train_reg_wf.columns,
                            )
                            X_test_scaled = pd.DataFrame(
                                wf_scaler.transform(X_test_reg_wf),
                                index=X_test_reg_wf.index,
                                columns=X_test_reg_wf.columns,
                            )
                            wf_lstm = train_lstm_model(
                                X_train=X_train_scaled,
                                y_train=y_train_reg_wf,
                                lookback=20,
                                hidden_size=32,
                                epochs=25,
                                learning_rate=0.001,
                                batch_size=32,
                                device="cpu",
                            )
                            preds = predict_lstm(
                                model_dict=wf_lstm,
                                X_test=X_test_scaled,
                                y_test=y_test_reg_wf,
                                device="cpu",
                            )
                            y_true = y_test_reg_wf.loc[preds.index]
                        else:
                            continue

                        mse = float(mean_squared_error(y_true, preds))
                        rmse = float(np.sqrt(mse))
                        directional_accuracy = float((np.sign(y_true) == np.sign(preds)).mean())
                        bt = run_backtest(
                            returns=fold_test.loc[y_true.index, "target_5d"],
                            signals=preds,
                            holding_period=args.holding_period,
                            transaction_cost=args.transaction_cost,
                            threshold=args.threshold,
                            model_name=model_name,
                        )
                        sharpe = float(bt.get("sharpe", np.nan))
                        mean_ic = (
                            float(pd.Series(y_true).corr(pd.Series(preds)))
                            if task_type == "regression"
                            else float("nan")
                        )
                        regime_eval = evaluate_by_regime(
                            df=fold_test.loc[y_true.index],
                            y_true=pd.Series(y_true, index=y_true.index),
                            y_pred=pd.Series(preds, index=preds.index),
                            model_name=model_name,
                        )
                        regime_dispersion = float(
                            pd.to_numeric(regime_eval.get("Directional_Accuracy"), errors="coerce").std(ddof=0)
                        ) if isinstance(regime_eval, pd.DataFrame) and not regime_eval.empty else float("nan")

                        fold_rows.append(
                            {
                                "fold": fold_label,
                                "model": model_name,
                                "task_type": task_type,
                                "mse": mse,
                                "rmse": rmse,
                                "directional_accuracy": directional_accuracy,
                                "sharpe": sharpe,
                                "mean_ic": mean_ic,
                                "regime_dispersion": regime_dispersion,
                            }
                        )
                    except Exception as wf_exc:
                        logger.warning("Walk-forward fold failed | fold=%s | model=%s | error=%s", fold_label, model_key, wf_exc)

        if fold_rows:
            walk_forward_fold_df = pd.DataFrame(fold_rows)
            grouped = (
                walk_forward_fold_df
                .groupby(["model", "task_type"], as_index=False)
                .mean(numeric_only=True)
            )
            walk_forward_comparison_df = grouped.rename(
                columns={
                    "model": "Model",
                    "task_type": "Task_Type",
                    "mse": "MSE",
                    "rmse": "RMSE",
                    "sharpe": "Sharpe",
                    "directional_accuracy": "Directional_Accuracy",
                    "mean_ic": "Mean_IC",
                    "regime_dispersion": "Regime_Dispersion",
                }
            )
            walk_forward_comparison_df["Composite_Score"] = compute_composite_score(
                walk_forward_comparison_df
            )
            walk_forward_comparison_df = walk_forward_comparison_df.sort_values(
                "Composite_Score", ascending=False, na_position="last"
            ).reset_index(drop=True)
            walk_forward_comparison_df["Rank"] = (
                walk_forward_comparison_df["Composite_Score"]
                .rank(ascending=False, method="dense")
                .astype("Int64")
            )
            logger.info("Walk-forward validation complete with %d folds.", walk_forward_fold_df["fold"].nunique())

    model_evaluations: list[dict[str, Any]] = []

    if "linear" in selected_models:
        linear_model = train_linear_regression(X_train=X_train_reg, y_train=y_train_reg)
        linear_preds = pd.Series(linear_model.predict(X_test_reg), index=X_test_reg.index)
        linear_eval = evaluate_regression(model=linear_model, X_test=X_test_reg, y_test=y_test_reg)
        overall_metrics = {
            "mse": float(linear_eval["model_mse"]),
            "rmse": float(linear_eval["model_rmse"]),
            "directional_accuracy": float(linear_eval["directional_accuracy"]),
        }
        model_evaluations.append(
            _build_model_evaluation(
                model_name="LinearRegression",
                task_type="regression",
                overall_metrics=overall_metrics,
                y_true=y_test_reg,
                y_pred=linear_preds,
                test_df=test_df,
                holding_period=args.holding_period,
                transaction_cost=args.transaction_cost,
                threshold=args.threshold,
            )
        )

    if "ridge" in selected_models:
        best_alpha, _ridge_results, ridge_model, ridge_scaler = train_ridge_model(
            X_train=X_train_reg,
            y_train=y_train_reg,
            X_test=X_test_reg,
            y_test=y_test_reg,
        )
        ridge_preds = pd.Series(
            ridge_model.predict(ridge_scaler.transform(X_test_reg)),
            index=X_test_reg.index,
        )
        overall_metrics = _regression_overall_metrics(y_true=y_test_reg, y_pred=ridge_preds)
        overall_metrics["best_alpha"] = float(best_alpha)
        model_evaluations.append(
            _build_model_evaluation(
                model_name="Ridge",
                task_type="regression",
                overall_metrics=overall_metrics,
                y_true=y_test_reg,
                y_pred=ridge_preds,
                test_df=test_df,
                holding_period=args.holding_period,
                transaction_cost=args.transaction_cost,
                threshold=args.threshold,
            )
        )

    if "random_forest" in selected_models:
        _rf_metrics, rf_model = train_random_forest_model(
            X_train=X_train_reg,
            y_train=y_train_reg,
            X_test=X_test_reg,
            y_test=y_test_reg,
        )
        rf_preds = pd.Series(rf_model.predict(X_test_reg), index=X_test_reg.index)
        overall_metrics = _regression_overall_metrics(y_true=y_test_reg, y_pred=rf_preds)
        model_evaluations.append(
            _build_model_evaluation(
                model_name="RandomForest",
                task_type="regression",
                overall_metrics=overall_metrics,
                y_true=y_test_reg,
                y_pred=rf_preds,
                test_df=test_df,
                holding_period=args.holding_period,
                transaction_cost=args.transaction_cost,
                threshold=args.threshold,
            )
        )

    if "lstm" in selected_models:
        logger.info("Training LSTM model.")
        lstm_scaler = StandardScaler()
        X_train_lstm_scaled = pd.DataFrame(
            lstm_scaler.fit_transform(X_train_reg),
            index=X_train_reg.index,
            columns=X_train_reg.columns,
        )
        X_test_lstm_scaled = pd.DataFrame(
            lstm_scaler.transform(X_test_reg),
            index=X_test_reg.index,
            columns=X_test_reg.columns,
        )

        lstm_dict = train_lstm_model(
            X_train=X_train_lstm_scaled,
            y_train=y_train_reg,
            lookback=20,
            hidden_size=32,
            epochs=25,
            learning_rate=0.001,
            batch_size=32,
            device="cpu",
        )
        lstm_preds = predict_lstm(
            model_dict=lstm_dict,
            X_test=X_test_lstm_scaled,
            y_test=y_test_reg,
            device="cpu",
        )
        y_test_lstm_aligned = y_test_reg.loc[lstm_preds.index]

        overall_metrics = _regression_overall_metrics(
            y_true=y_test_lstm_aligned,
            y_pred=lstm_preds,
        )
        model_evaluations.append(
            _build_model_evaluation(
                model_name="LSTM",
                task_type="regression",
                overall_metrics=overall_metrics,
                y_true=y_test_lstm_aligned,
                y_pred=lstm_preds,
                test_df=test_df,
                holding_period=args.holding_period,
                transaction_cost=args.transaction_cost,
                threshold=args.threshold,
            )
        )

    # Safety filter: enforce strict 4-model research scope before aggregation/plotting.
    model_results = {m["model_name"]: m for m in model_evaluations if m.get("model_name") in ALLOWED_MODELS}
    model_evaluations = list(model_results.values())

    if args.use_walk_forward and not walk_forward_comparison_df.empty:
        comparison_df = walk_forward_comparison_df[
            walk_forward_comparison_df["Model"].isin(ALLOWED_MODELS)
        ].copy()
    else:
        comparison_df = compare_models(model_evaluations)
    comparison_df = comparison_df[comparison_df["Model"].isin(ALLOWED_MODELS)].copy()
    best_summary = summarize_best_model(comparison_df)
    _print_comparison(comparison_df=comparison_df, best_summary=best_summary)
    if args.advanced_analysis:
        _print_advanced_diagnostics(model_evaluations=model_evaluations)

    _random_returns, _random_equity, random_sharpe = _compute_random_baseline(test_df=test_df)
    _print_market_efficiency_insight(
        model_evaluations=model_evaluations,
        best_model_name=str(best_summary.get("best_model", "")),
        random_sharpe=random_sharpe,
    )
    best_sharpe_val = float(best_summary.get("sharpe", np.nan))
    if np.isfinite(best_sharpe_val) and np.isfinite(random_sharpe) and (best_sharpe_val - random_sharpe) > 1.0:
        print("Check for leakage or overfitting")

    if args.save_results:
        experiments_root = Path(__file__).resolve().parent / "experiments"
        results_file = save_run_results(
            base_dir=experiments_root,
            ticker=args.ticker.strip().upper(),
            start=args.start,
            end=args.end,
            dataset_version=dataset_version,
            comparison_df=comparison_df,
        )
        leaderboard_file = append_leaderboard(
            base_dir=experiments_root,
            ticker=args.ticker.strip().upper(),
            start=args.start,
            end=args.end,
            best_model=str(best_summary.get("best_model", "")),
            sharpe=float(best_summary.get("sharpe", np.nan)),
            composite_score=float(best_summary.get("score", np.nan)),
            data_source=data_source,
        )
        run_log_file = append_run_log(
            base_dir=experiments_root,
            payload={
                "ticker": args.ticker.strip().upper(),
                "start": args.start,
                "end": args.end,
                "models_used": selected_models,
                "dataset_version": dataset_version,
                "data_source": data_source,
                "synthetic_flag": is_synthetic,
                "use_walk_forward": bool(args.use_walk_forward),
                "folds": int(walk_forward_fold_df["fold"].nunique()) if not walk_forward_fold_df.empty else 0,
            },
        )
        logger.info("Saved run results: %s", results_file)
        if not walk_forward_fold_df.empty:
            fold_file = (
                experiments_root
                / "results"
                / f"{args.ticker.strip().upper()}_{args.start}_{args.end}_v{dataset_version}_folds.csv"
            )
            fold_file.parent.mkdir(parents=True, exist_ok=True)
            walk_forward_fold_df.to_csv(fold_file, index=False)
            logger.info("Saved walk-forward fold results: %s", fold_file)
        logger.info("Updated leaderboard: %s", leaderboard_file)
        logger.info("Updated run log: %s", run_log_file)

    if args.analysis_mode == "auto" and not args.no_plots:
        _plot_auto_analysis(
            model_evaluations=model_evaluations,
            logger=logger,
            test_df=test_df,
            best_model_name=str(best_summary.get("best_model", "")),
            ticker=args.ticker,
        )


if __name__ == "__main__":
    main()
