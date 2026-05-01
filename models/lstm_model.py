"""LSTM regression model utilities for forward-return prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    """Sequence-to-one LSTM regressor."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize LSTM regressor architecture.

        Args:
            input_size: Number of features per timestep.
            hidden_size: Hidden state size of LSTM.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout between stacked LSTM layers.
        """
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Tensor of shape (batch_size, 1).
        """
        if x.ndim != 3:
            raise ValueError("Input x must be 3D: (batch_size, sequence_length, input_size).")
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.output_layer(last_hidden)


def _resolve_device(device: str) -> torch.device:
    """Resolve torch device with safe fallback to CPU."""
    normalized = device.strip().lower()
    if normalized == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Convert tabular features into rolling LSTM sequences.

    For each target timestamp ``t``, the input window is ``X[t-lookback : t]`` and
    the target is ``y[t]``.

    Args:
        X: Feature DataFrame in chronological order.
        y: Target series aligned to ``X`` index.
        lookback: Number of historical timesteps in each sequence.

    Returns:
        X_seq: Array of shape (num_samples, lookback, num_features).
        y_seq: Array of shape (num_samples,).
        index_seq: Target index aligned with sequence outputs.

    Raises:
        TypeError: If inputs are not pandas objects.
        ValueError: If lookback or alignment constraints are violated.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    if lookback <= 0:
        raise ValueError("lookback must be a positive integer.")
    if X.empty or y.empty:
        raise ValueError("X and y must be non-empty.")
    if not X.index.equals(y.index):
        raise ValueError("X and y indices must match exactly.")
    if lookback >= len(X):
        raise ValueError("lookback must be smaller than the number of rows in X.")

    X_values = X.to_numpy(dtype=np.float32, copy=True)
    y_values = y.to_numpy(dtype=np.float32, copy=True)

    n_samples = len(X_values) - lookback
    n_features = X_values.shape[1]

    X_seq = np.empty((n_samples, lookback, n_features), dtype=np.float32)
    y_seq = np.empty(n_samples, dtype=np.float32)
    index_list = []

    for t in range(lookback, len(X_values)):
        seq_idx = t - lookback
        X_seq[seq_idx] = X_values[t - lookback : t]
        y_seq[seq_idx] = y_values[t]
        index_list.append(X.index[t])

    index_seq = pd.Index(index_list)
    return X_seq, y_seq, index_seq


def train_lstm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lookback: int = 20,
    hidden_size: int = 32,
    num_layers: int = 1,
    epochs: int = 25,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train an LSTM regressor on rolling feature sequences.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target series aligned with ``X_train``.
        lookback: Number of timesteps per sequence window.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        device: ``"cpu"`` or ``"cuda"`` (uses CUDA only if available).

    Returns:
        Dictionary containing:
        - ``model``: trained ``LSTMRegressor``
        - ``lookback``: sequence lookback used
        - ``input_size``: number of input features

    Raises:
        ValueError: If training hyperparameters are invalid.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # Reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    resolved_device = _resolve_device(device)
    X_seq, y_seq, _ = create_sequences(X=X_train, y=y_train, lookback=lookback)

    input_size = int(X_seq.shape[2])
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(resolved_device)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=resolved_device)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32, device=resolved_device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x).squeeze(-1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            batch_count = batch_x.size(0)
            epoch_loss_sum += float(loss.item()) * batch_count
            total_samples += batch_count

        epoch_loss = epoch_loss_sum / total_samples if total_samples > 0 else float("nan")
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.8f}")

    return {
        "model": model,
        "lookback": int(lookback),
        "input_size": input_size,
    }


def predict_lstm(
    model_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    device: str = "cpu",
) -> pd.Series:
    """Generate aligned out-of-sample predictions from a trained LSTM model.

    Args:
        model_dict: Output dictionary from ``train_lstm_model``.
        X_test: Test feature DataFrame.
        y_test: Test target series used for alignment and sequence construction.
        device: ``"cpu"`` or ``"cuda"`` (uses CUDA only if available).

    Returns:
        Prediction series aligned to sequence target index.

    Raises:
        ValueError: If required model metadata is missing or dimensionality mismatches.
        TypeError: If model object is invalid.
    """
    if not isinstance(model_dict, dict):
        raise TypeError("model_dict must be a dictionary.")

    required_keys = {"model", "lookback", "input_size"}
    missing_keys = required_keys.difference(model_dict.keys())
    if missing_keys:
        raise ValueError(f"model_dict is missing required keys: {sorted(missing_keys)}")

    model = model_dict["model"]
    if not isinstance(model, nn.Module):
        raise TypeError("model_dict['model'] must be a torch.nn.Module.")

    lookback = int(model_dict["lookback"])
    input_size = int(model_dict["input_size"])
    if X_test.shape[1] != input_size:
        raise ValueError(
            f"Feature dimension mismatch. Expected {input_size}, got {X_test.shape[1]}."
        )

    resolved_device = _resolve_device(device)
    X_seq, _y_seq, index_seq = create_sequences(X=X_test, y=y_test, lookback=lookback)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=resolved_device)

    model = model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).squeeze(-1).detach().cpu().numpy()

    return pd.Series(preds, index=index_seq, name="lstm_prediction")
