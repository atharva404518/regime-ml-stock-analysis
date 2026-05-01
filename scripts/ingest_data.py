"""CLI script for standalone data ingestion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import setup_logging
from data.dataset_manager import save_versioned_dataset
from data.loader import load_market_data_with_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest market data into versioned datasets.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data_mode", choices=["auto", "tiingo", "local", "synthetic"], default="auto")
    parser.add_argument("--save_dataset", action="store_true")
    parser.add_argument("--quality_report", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    df, report = load_market_data_with_report(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        mode=args.data_mode,
    )

    logger.info("Ingestion completed | source=%s | rows=%d", df.attrs.get("source"), len(df))

    if args.save_dataset:
        record = save_versioned_dataset(
            df=df,
            ticker=args.ticker.upper(),
            start=args.start,
            end=args.end,
            source=str(df.attrs.get("source", "unknown")),
            datasets_root=Path(__file__).resolve().parents[1] / "data" / "datasets",
        )
        logger.info("Saved dataset version: %s", record)

    if args.quality_report:
        logger.info("Quality Report: %s", report)


if __name__ == "__main__":
    main()
