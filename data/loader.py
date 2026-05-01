"""Primary ingestion loader orchestrating multi-source fallback."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.sources.local_csv import load_local_csv_ohlcv
from data.sources.synthetic import generate_synthetic_ohlcv
from data.sources.tiingo import fetch_tiingo_ohlcv
from data.validation.quality_report import generate_quality_report
from data.validation.validator import validate_ohlcv_dataframe

LOGGER = logging.getLogger(__name__)

_DATASETS_ROOT = Path(__file__).resolve().parent / "datasets"
_RAW_DIR = _DATASETS_ROOT / "raw"


def _with_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out.attrs["source"] = source
    metadata = dict(out.attrs.get("dataset_metadata", {}))
    metadata["is_synthetic"] = bool(source == "synthetic")
    metadata["source"] = source
    out.attrs["dataset_metadata"] = metadata
    if source == "synthetic":
        LOGGER.warning(
            "WARNING: Synthetic data is being used. Results are NOT valid for research conclusions."
        )
    return out


def load_market_data(
    ticker: str,
    start: str,
    end: str,
    mode: str = "auto",
    missing_threshold: float = 0.20,
    max_retries: int = 5,
    timeout: int = 30,
) -> pd.DataFrame:
    """Load market OHLCV with robust source hierarchy and fail-safe fallback.

    Auto hierarchy:
        tiingo -> local csv -> synthetic
    """
    ticker = ticker.strip().upper()
    if ".NS" in ticker:
        raise ValueError("NSE tickers not supported. Use US equities or crypto.")
    selected_mode = (mode or "auto").strip().lower()
    if selected_mode == "api":
        selected_mode = "tiingo"
    if selected_mode == "cache":
        selected_mode = "local"

    LOGGER.info(
        "Data loader started | ticker=%s | start=%s | end=%s | mode=%s",
        ticker,
        start,
        end,
        selected_mode,
    )

    if selected_mode == "synthetic":
        df = generate_synthetic_ohlcv(start=start, end=end)
        df = validate_ohlcv_dataframe(df, missing_threshold=missing_threshold)
        LOGGER.warning("Data source selected: synthetic (forced mode)")
        return _with_source(df, "synthetic")

    if selected_mode == "local":
        try:
            df = load_local_csv_ohlcv(ticker=ticker, start=start, end=end, base_dir=_RAW_DIR)
            df = validate_ohlcv_dataframe(df, missing_threshold=missing_threshold)
            LOGGER.info("Data source selected: local")
            return _with_source(df, "local")
        except Exception as exc:
            LOGGER.warning("Local mode failed; falling back to synthetic: %s", exc)
            df = generate_synthetic_ohlcv(start=start, end=end)
            df = validate_ohlcv_dataframe(df, missing_threshold=missing_threshold)
            return _with_source(df, "synthetic")

    if selected_mode == "tiingo":
        try:
            df = fetch_tiingo_ohlcv(
                ticker=ticker,
                start=start,
                end=end,
                timeout=timeout,
                max_retries=max_retries,
            )
            df = validate_ohlcv_dataframe(df, missing_threshold=missing_threshold)
            LOGGER.info("Data source selected: api (tiingo)")
            return _with_source(df, "api")
        except Exception as exc:
            LOGGER.warning("Tiingo mode failed; falling back to synthetic: %s", exc)
            df = generate_synthetic_ohlcv(start=start, end=end)
            df = validate_ohlcv_dataframe(df, missing_threshold=missing_threshold)
            return _with_source(df, "synthetic")

    # Auto mode hierarchy: tiingo -> local -> synthetic.
    try:
        LOGGER.info("Auto mode step 1: trying Tiingo API")
        df = fetch_tiingo_ohlcv(
            ticker=ticker,
            start=start,
            end=end,
            timeout=timeout,
            max_retries=max_retries,
        )
        df = validate_ohlcv_dataframe(df, missing_threshold=missing_threshold)
        LOGGER.info("Final source used: api (tiingo)")
        return _with_source(df, "api")
    except Exception as api_exc:
        LOGGER.warning("Auto mode Tiingo step failed: %s", api_exc)

    try:
        LOGGER.info("Auto mode step 2: trying local CSV")
        df = load_local_csv_ohlcv(ticker=ticker, start=start, end=end, base_dir=_RAW_DIR)
        df = validate_ohlcv_dataframe(df, missing_threshold=missing_threshold)
        LOGGER.info("Final source used: local")
        return _with_source(df, "local")
    except Exception as local_exc:
        LOGGER.warning("Auto mode local CSV step failed: %s", local_exc)

    LOGGER.warning("Auto mode step 3: using synthetic fail-safe data")
    df = generate_synthetic_ohlcv(start=start, end=end)
    df = validate_ohlcv_dataframe(df, missing_threshold=missing_threshold)
    return _with_source(df, "synthetic")


def load_market_data_with_report(
    ticker: str,
    start: str,
    end: str,
    mode: str = "auto",
    missing_threshold: float = 0.20,
    max_retries: int = 5,
    timeout: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Load data and return accompanying quality report."""
    df = load_market_data(
        ticker=ticker,
        start=start,
        end=end,
        mode=mode,
        missing_threshold=missing_threshold,
        max_retries=max_retries,
        timeout=timeout,
    )
    report = generate_quality_report(df)
    report["source"] = str(df.attrs.get("source", "unknown"))
    report["is_synthetic"] = bool(df.attrs.get("dataset_metadata", {}).get("is_synthetic", False))
    return df, report

