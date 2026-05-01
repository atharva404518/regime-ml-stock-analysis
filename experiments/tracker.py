"""Experiment tracking utilities for quantitative research runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _paths(base_dir: Path) -> dict[str, Path]:
    results_dir = base_dir / "results"
    summary_dir = base_dir / "summary"
    metadata_dir = base_dir / "metadata"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": results_dir,
        "leaderboard": summary_dir / "leaderboard.csv",
        "run_log": metadata_dir / "run_log.json",
    }


def save_run_results(
    base_dir: Path,
    ticker: str,
    start: str,
    end: str,
    dataset_version: str,
    comparison_df: pd.DataFrame,
) -> Path:
    """Save per-run model metrics table."""
    p = _paths(base_dir)
    out_file = p["results_dir"] / f"{ticker}_{start}_{end}_v{dataset_version}.csv"

    export_df = pd.DataFrame(
        {
            "model": comparison_df.get("Model"),
            "mse": comparison_df.get("MSE"),
            "rmse": comparison_df.get("RMSE"),
            "sharpe": comparison_df.get("Sharpe"),
            "directional_accuracy": comparison_df.get("Directional_Accuracy"),
            "mean_ic": comparison_df.get("Mean_IC"),
            "regime_dispersion": comparison_df.get("Regime_Dispersion"),
        }
    )
    export_df.to_csv(out_file, index=False)
    return out_file


def append_leaderboard(
    base_dir: Path,
    ticker: str,
    start: str,
    end: str,
    best_model: str,
    sharpe: float,
    composite_score: float,
    data_source: str,
) -> Path:
    """Append a run summary row to global leaderboard."""
    p = _paths(base_dir)
    leaderboard = p["leaderboard"]
    row = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "period": f"{start} -> {end}",
                "best_model": best_model,
                "sharpe": sharpe,
                "composite_score": composite_score,
                "data_source": data_source,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    if leaderboard.exists():
        existing = pd.read_csv(leaderboard)
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(leaderboard, index=False)
    return leaderboard


def append_run_log(base_dir: Path, payload: dict) -> Path:
    """Append structured JSON metadata for a run."""
    p = _paths(base_dir)
    run_log = p["run_log"]
    if run_log.exists():
        try:
            data = json.loads(run_log.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    payload = dict(payload)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    data.append(payload)
    run_log.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return run_log
