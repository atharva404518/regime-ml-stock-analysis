"""Baseline modeling utilities for time-series regression."""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def _annualized_sharpe(strategy_returns: pd.Series) -> float:
    """Compute annualized Sharpe ratio with safe handling."""
    ret = pd.to_numeric(strategy_returns, errors="coerce").dropna()
    if ret.empty:
        return float("-inf")
    std = float(ret.std(ddof=0))
    if std <= 0:
        return float("-inf")
    return float((ret.mean() / std) * np.sqrt(252.0))


def _time_split_index(n_samples: int, train_fraction: float = 0.8) -> int:
    """Return chronological split index with safeguards."""
    split_idx = int(n_samples * train_fraction)
    return max(20, min(split_idx, n_samples - 20))


def train_test_split_time_series(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame into chronological train and test sets.

    Args:
        df: Input DataFrame sorted in chronological order.
        test_size: Fraction of rows assigned to the test set.

    Returns:
        Tuple of ``(train_df, test_df)``.

    Raises:
        ValueError: If ``test_size`` would create an empty split.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Invalid split: train or test set would be empty.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    LOGGER.info(
        "Time-series split complete | train rows: %d | test rows: %d",
        len(train_df),
        len(test_df),
    )
    return train_df, test_df


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Backward-compatible feature/target preparation for regression tasks."""
    return prepare_xy_regression(df)

def prepare_xy_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and continuous 5-day return target."""
    if "target_5d" not in df.columns:
        raise ValueError("Target column 'target_5d' not found.")

    exclude_cols = {
        "target",
        "target_5d",
        "target_5d_class",
        "Trend_Regime",
        "Vol_Regime",
    }

    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df["target_5d"]

    raw_price_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    X = X.drop(columns=raw_price_cols, errors="ignore")

    return X, y

def prepare_xy_classification(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and binary directional target."""
    if "target_5d_class" not in df.columns:
        raise ValueError("Target column 'target_5d_class' not found.")

    exclude_cols = {
        "target",
        "target_5d",
        "target_5d_class",
        "Trend_Regime",
        "Vol_Regime",
    }

    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df["target_5d_class"]

    raw_price_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    X = X.drop(columns=raw_price_cols, errors="ignore")

    return X, y

def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """Train a baseline linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    LOGGER.info("LinearRegression training complete on %d samples.", len(X_train))
    return model


def train_ridge_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Train and evaluate Ridge models across alpha values on scaled features.

    Scaler is fit on train data only to avoid data leakage.
    """
    alpha_values = [0.1, 1.0, 10.0, 100.0]
    results = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_alpha = None
    best_model = None
    best_mse = np.inf

    for alpha in alpha_values:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train_scaled, y_train)
        preds = ridge_model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        directional_accuracy = float((np.sign(preds) == np.sign(y_test)).mean())

        results[alpha] = {
            "mse": float(mse),
            "rmse": rmse,
            "directional_accuracy": directional_accuracy,
        }
        LOGGER.info(
            "Ridge alpha=%.1f | MSE: %.8f | RMSE: %.8f | Directional Accuracy: %.4f",
            alpha,
            results[alpha]["mse"],
            results[alpha]["rmse"],
            results[alpha]["directional_accuracy"],
        )

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_model = ridge_model

    LOGGER.info("Best Ridge alpha selected: %.1f (MSE: %.8f)", best_alpha, best_mse)
    return best_alpha, results, best_model, scaler


def train_random_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Train and evaluate a Random Forest regressor for time-series prediction."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    directional_accuracy = float((np.sign(preds) == np.sign(y_test)).mean())

    metrics = {
        "mse": float(mse),
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
    }
    LOGGER.info(
        "Random Forest | MSE: %.8f | RMSE: %.8f | Directional Accuracy: %.4f",
        metrics["mse"],
        metrics["rmse"],
        metrics["directional_accuracy"],
    )
    return metrics, model


def train_logistic_model(
    X_train,
    y_train,
    X_test,
    y_test,
    train_returns: pd.Series | None = None,
    test_returns: pd.Series | None = None,
):
    """Train and evaluate calibrated Logistic Regression with threshold tuning."""
    if len(X_train) < 40:
        raise ValueError("Logistic model requires at least 40 training samples.")

    if train_returns is not None:
        train_returns = pd.to_numeric(train_returns, errors="coerce").reindex(X_train.index)
    if test_returns is not None:
        test_returns = pd.to_numeric(test_returns, errors="coerce").reindex(X_test.index)

    split_idx = _time_split_index(len(X_train), train_fraction=0.8)
    X_subtrain = X_train.iloc[:split_idx]
    y_subtrain = y_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]
    if train_returns is not None:
        val_returns = train_returns.iloc[split_idx:]
    else:
        val_returns = pd.Series(
            np.where(y_val.to_numpy(dtype=float) > 0.5, 1.0, -1.0),
            index=y_val.index,
            dtype=float,
        )

    if y_subtrain.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError("Logistic model requires both classes in train/validation splits.")

    scaler = StandardScaler()
    X_subtrain_scaled = scaler.fit_transform(X_subtrain)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    base_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    base_model.fit(X_subtrain_scaled, y_subtrain)

    # Platt calibration on validation set probabilities.
    val_prob_raw = base_model.predict_proba(X_val_scaled)[:, 1]
    platt_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    platt_model.fit(val_prob_raw.reshape(-1, 1), y_val)
    val_prob_calibrated = platt_model.predict_proba(val_prob_raw.reshape(-1, 1))[:, 1]

    # Threshold tuning (0.50 -> 0.70), optimized by validation Sharpe.
    threshold_grid = np.arange(0.50, 0.701, 0.02)
    best_threshold = 0.50
    best_sharpe = float("-inf")
    for threshold in threshold_grid:
        val_signal = pd.Series(
            np.where(val_prob_calibrated >= threshold, 1.0, 0.0),
            index=y_val.index,
            dtype=float,
        )
        val_strategy_returns = val_signal * val_returns
        sharpe = _annualized_sharpe(val_strategy_returns)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = float(threshold)

    # Refit base classifier on full train, then apply fitted calibrator.
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)
    test_prob_raw = model.predict_proba(X_test_scaled)[:, 1]
    test_prob_calibrated = platt_model.predict_proba(test_prob_raw.reshape(-1, 1))[:, 1]
    preds = (test_prob_calibrated >= best_threshold).astype(int)

    majority_class = y_train.mode()[0]
    baseline_preds = np.full(shape=len(y_test), fill_value=majority_class)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)
    train_class_distribution = y_train.value_counts().to_dict()
    test_class_distribution = y_test.value_counts().to_dict()
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "baseline_accuracy": float(baseline_accuracy),
        "train_class_distribution": {
            0: int(train_class_distribution.get(0, 0)),
            1: int(train_class_distribution.get(1, 0)),
        },
        "test_class_distribution": {
            0: int(test_class_distribution.get(0, 0)),
            1: int(test_class_distribution.get(1, 0)),
        },
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "best_threshold": float(best_threshold),
        "threshold_tuning_metric": "validation_sharpe",
        "validation_sharpe": float(best_sharpe),
        "calibration_method": "platt",
    }
    LOGGER.info(
        "Logistic | Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | "
        "Baseline Accuracy: %.4f | TN: %d | FP: %d | FN: %d | TP: %d | "
        "Best Threshold: %.2f | Validation Sharpe: %.4f | Calibration: %s",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["baseline_accuracy"],
        metrics["tn"],
        metrics["fp"],
        metrics["fn"],
        metrics["tp"],
        metrics["best_threshold"],
        metrics["validation_sharpe"],
        metrics["calibration_method"],
    )
    return metrics, model, scaler, platt_model


def evaluate_regression(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate regression performance using error and directional metrics."""
    preds = model.predict(X_test)
    naive_preds = np.zeros_like(y_test)

    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    naive_mse = mean_squared_error(y_test, naive_preds)
    naive_rmse = float(np.sqrt(naive_mse))
    directional_accuracy = float((np.sign(preds) == np.sign(y_test)).mean())

    metrics = {
        "model_mse": float(mse),
        "model_rmse": float(rmse),
        "naive_mse": float(naive_mse),
        "naive_rmse": float(naive_rmse),
        "directional_accuracy": float(directional_accuracy),
    }
    LOGGER.info(
        "Evaluation | Model MSE: %.8f | Naive MSE: %.8f | "
        "Model RMSE: %.8f | Naive RMSE: %.8f | Directional Accuracy: %.4f",
        metrics["model_mse"],
        metrics["naive_mse"],
        metrics["model_rmse"],
        metrics["naive_rmse"],
        metrics["directional_accuracy"],
    )
    return metrics

def naive_lag_signal(y_test: pd.Series) -> pd.Series:
    """
    Predict next return = previous return (lag-1 persistence).
    """
    return lag_return_baseline(y_test, lag=1)


def lag_return_baseline(returns: pd.Series, lag: int = 1) -> pd.Series:
    """Lag-return baseline: prediction at t equals realized return at t-lag."""
    if lag <= 0:
        raise ValueError("lag must be a positive integer.")
    preds = pd.to_numeric(returns, errors="coerce").shift(lag).fillna(0.0)
    LOGGER.info("Lag-return baseline generated | lag=%d | samples=%d", lag, len(preds))
    return preds.astype(float)

def momentum_signal(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Simple momentum: rolling mean of past returns.
    """
    if "target_5d" not in df.columns:
        raise ValueError("target_5d required for momentum baseline.")
    return momentum_baseline(df["target_5d"], window=window)


def momentum_baseline(returns: pd.Series, window: int = 5) -> pd.Series:
    """Momentum baseline: rolling mean of past returns, shifted for anti-leakage."""
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    momentum = pd.to_numeric(returns, errors="coerce").rolling(window=window).mean()
    preds = momentum.shift(1).fillna(0.0)
    LOGGER.info("Momentum baseline generated | window=%d | samples=%d", window, len(preds))
    return preds.astype(float)

def moving_average_crossover_signal(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
) -> pd.Series:
    """
    Generate signal based on MA crossover.
    Long when short MA > long MA.
    """
    if "Close" not in df.columns:
        raise ValueError("Close price required for MA crossover.")

    return moving_average_signal(
        df=df,
        short_window=short_window,
        long_window=long_window,
    )


def moving_average_signal(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    price_col: str = "Close",
) -> pd.Series:
    """Continuous MA signal: normalized MA spread, shifted to avoid look-ahead."""
    if short_window <= 0 or long_window <= 0:
        raise ValueError("short_window and long_window must be positive integers.")
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window.")
    if price_col not in df.columns:
        raise ValueError(f"{price_col} price required for MA baseline.")

    price = pd.to_numeric(df[price_col], errors="coerce")
    short_ma = price.rolling(short_window).mean()
    long_ma = price.rolling(long_window).mean()
    signal = ((short_ma / long_ma) - 1.0).replace([np.inf, -np.inf], np.nan)
    preds = signal.shift(1).fillna(0.0)
    LOGGER.info(
        "MA baseline generated | short=%d | long=%d | samples=%d",
        short_window,
        long_window,
        len(preds),
    )
    return preds.astype(float)

