"""Download and cache local OHLCV CSV files via Tiingo."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from config import setup_logging
from data.sources.tiingo import fetch_tiingo_ohlcv
from data.validation.validator import validate_ohlcv_dataframe

LOGGER = logging.getLogger(__name__)

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "BTCUSD"]


def _is_crypto(ticker: str) -> bool:
    symbol = ticker.strip().upper()
    return symbol.endswith("USD") and len(symbol) >= 6


def _fill_missing_dates(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Reindex to continuous date grid and forward-fill OHLCV."""
    freq = "D" if _is_crypto(ticker) else "B"
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    out = df.reindex(full_index).sort_index()
    out = out.ffill().bfill()
    out.index.name = "Date"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Tiingo data into local CSV files.")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    local_dir = Path(__file__).resolve().parents[1] / "data" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    for ticker in [t.strip().upper() for t in args.tickers if t.strip()]:
        try:
            LOGGER.info("Downloading %s from Tiingo.", ticker)
            df = fetch_tiingo_ohlcv(ticker=ticker, start=args.start, end=args.end)
            df = df[~df.index.duplicated(keep="first")].sort_index()
            df = _fill_missing_dates(df, ticker=ticker)
            df = validate_ohlcv_dataframe(df, missing_threshold=0.20)
            output_path = local_dir / f"{ticker}.csv"
            df.to_csv(output_path, index_label="Date")
            LOGGER.info("Saved local dataset: %s (%d rows)", output_path, len(df))
        except Exception as exc:
            LOGGER.error("Failed to download %s: %s", ticker, exc)


if __name__ == "__main__":
    main()
