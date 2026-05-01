"""Rolling performance metrics for time-series model evaluation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def _validate_inputs(y_true: pd.Series, y_pred: pd.Series, window: int) -> None:
    """Validate rolling metric input constraints."""
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        raise TypeError("y_true and y_pred must be pandas Series.")
    if not y_true.index.equals(y_pred.index):
        raise ValueError("Indices of y_true and y_pred must match.")
    if window < 2:
        raise ValueError("Window must be >= 2.")
    if window > len(y_true):
        raise ValueError("Window cannot exceed series length.")


def compute_rolling_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 60,
    task_type: Literal["regression", "classification"] = "regression",
    model_name: str = "Model",
) -> pd.DataFrame:
    """Compute rolling evaluation metrics over a fixed chronological window."""
    _validate_inputs(y_true, y_pred, window)

    results: list[dict[str, float | int | str | pd.Timestamp]] = []
    for end_idx in range(window, len(y_true) + 1):
        start_idx = end_idx - window
        y_true_window = y_true.iloc[start_idx:end_idx]
        y_pred_window = y_pred.iloc[start_idx:end_idx]

        row: dict[str, float | int | str | pd.Timestamp] = {
            "Date": y_true_window.index[-1],
            "Model": model_name,
            "Window": window,
        }

        if task_type == "regression":
            mse = mean_squared_error(y_true_window, y_pred_window)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_window, y_pred_window)
            directional_accuracy = np.mean(np.sign(y_true_window) == np.sign(y_pred_window))
            ic = np.corrcoef(y_true_window, y_pred_window)[0, 1]

            row.update(
                {
                    "Rolling_MSE": float(mse),
                    "Rolling_RMSE": float(rmse),
                    "Rolling_MAE": float(mae),
                    "Rolling_Directional_Accuracy": float(directional_accuracy),
                    "Rolling_IC": float(ic),
                }
            )
        elif task_type == "classification":
            acc = accuracy_score(y_true_window, y_pred_window)
            precision = precision_score(y_true_window, y_pred_window, zero_division=0)
            recall = recall_score(y_true_window, y_pred_window, zero_division=0)
            f1 = f1_score(y_true_window, y_pred_window, zero_division=0)

            row.update(
                {
                    "Rolling_Accuracy": float(acc),
                    "Rolling_Precision": float(precision),
                    "Rolling_Recall": float(recall),
                    "Rolling_F1": float(f1),
                }
            )
        else:
            raise ValueError("task_type must be 'regression' or 'classification'.")

        results.append(row)

    metrics_df = pd.DataFrame(results)
    metrics_df.set_index("Date", inplace=True)
    return metrics_df
