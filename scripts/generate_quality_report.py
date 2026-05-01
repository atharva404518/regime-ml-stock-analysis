"""Generate quality report for ingested datasets."""

from __future__ import annotations

import argparse
import logging

from config import setup_logging
from data.loader import load_market_data_with_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data quality report.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data_mode", choices=["auto", "tiingo", "local", "synthetic"], default="auto")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    _df, report = load_market_data_with_report(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        mode=args.data_mode,
    )
    logger.info("Quality report generated: %s", report)


if __name__ == "__main__":
    main()
