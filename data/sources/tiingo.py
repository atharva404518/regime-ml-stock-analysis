"""Tiingo source adapter for OHLCV ingestion."""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

_TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"
_TIINGO_CRYPTO_URL = "https://api.tiingo.com/tiingo/crypto/prices"
_USER_AGENT = "Mozilla/5.0"
_DEFAULT_CRYPTO_TICKERS = {"BTCUSD", "ETHUSD"}


def _load_env_value(key: str) -> str | None:
    """Load API key from process environment or project .env."""
    env_value = os.getenv(key)
    if env_value:
        return env_value.strip()

    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.exists():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        name, value = clean.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _build_session() -> requests.Session:
    """Build hardened requests session for Tiingo API calls."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
    )

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _normalize_tiingo_payload(payload: list[dict], ticker: str) -> pd.DataFrame:
    """Normalize Tiingo payload into canonical OHLCV DataFrame."""
    if not payload:
        raise ValueError("Tiingo returned empty payload.")

    df = pd.DataFrame(payload)
    if df.empty:
        raise ValueError("Tiingo returned empty dataframe.")

    if "date" not in df.columns:
        raise ValueError("Tiingo payload missing 'date' field.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[~df["date"].isna()].copy()
    df = df.set_index("date").sort_index()

    volume_col = "volume" if "volume" in df.columns else "adjVolume"
    required = ["open", "high", "low", "close", volume_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Tiingo payload missing columns: {missing}")

    out = df[["open", "high", "low", "close", volume_col]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="first")]

    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    if out.empty:
        raise ValueError(f"Tiingo normalized dataframe is empty for {ticker}.")
    return out


def _normalize_tiingo_crypto_payload(payload: list[dict], ticker: str) -> pd.DataFrame:
    """Normalize Tiingo crypto payload into canonical OHLCV DataFrame."""
    if not payload:
        raise ValueError("Tiingo crypto returned empty payload.")

    first = payload[0]
    if not isinstance(first, dict) or "priceData" not in first:
        raise ValueError("Tiingo crypto payload format unexpected.")

    price_data = first.get("priceData")
    if not isinstance(price_data, list) or not price_data:
        raise ValueError("Tiingo crypto priceData is empty.")

    df = pd.DataFrame(price_data)
    if "date" not in df.columns:
        raise ValueError("Tiingo crypto payload missing 'date' field.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[~df["date"].isna()].copy()
    df = df.set_index("date").sort_index()

    open_col = "open" if "open" in df.columns else "openPrice"
    high_col = "high" if "high" in df.columns else "highPrice"
    low_col = "low" if "low" in df.columns else "lowPrice"
    close_col = "close" if "close" in df.columns else "closePrice"
    volume_col = "volume" if "volume" in df.columns else "volumeNotional"

    required = [open_col, high_col, low_col, close_col, volume_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Tiingo crypto payload missing columns: {missing}")

    out = df[[open_col, high_col, low_col, close_col, volume_col]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="first")]

    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    if out.empty:
        raise ValueError(f"Tiingo crypto normalized dataframe is empty for {ticker}.")
    return out


def _is_crypto_ticker(ticker: str) -> bool:
    """Detect common Tiingo crypto ticker formats (e.g., BTCUSD, ETHUSD)."""
    normalized = ticker.strip().upper()
    if normalized in _DEFAULT_CRYPTO_TICKERS:
        return True
    return len(normalized) >= 6 and normalized.endswith("USD") and normalized[:-3].isalpha()


def fetch_tiingo_ohlcv(
    ticker: str,
    start: str,
    end: str,
    timeout: int = 30,
    max_retries: int = 5,
    base_delay: float = 3.0,
) -> pd.DataFrame:
    """Fetch OHLCV from Tiingo with retry/backoff and jitter."""
    api_key = _load_env_value("TINGO_API_KEY") or _load_env_value("TIINGO_API_KEY")
    if not api_key:
        raise ValueError("Tiingo API key not found in environment or .env.")

    ticker = ticker.strip().upper()
    if ".NS" in ticker:
        raise ValueError("NSE tickers not supported. Use US equities or crypto.")

    is_crypto = _is_crypto_ticker(ticker)
    if is_crypto:
        url = _TIINGO_CRYPTO_URL
        params = {
            "tickers": ticker,
            "startDate": start,
            "endDate": end,
            "resampleFreq": "1day",
            "token": api_key,
        }
    else:
        url = f"{_TIINGO_BASE_URL}/{ticker}/prices"
        params = {
            "startDate": start,
            "endDate": end,
            "resampleFreq": "daily",
            "token": api_key,
        }

    session = _build_session()
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        attempt_no = attempt + 1
        try:
            LOGGER.info("Tiingo request | ticker=%s | attempt=%d/%d", ticker, attempt_no, max_retries)
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            text_preview = response.text[:200].lower()
            if "text/html" in content_type or "<html" in text_preview:
                raise ValueError("Tiingo returned HTML response.")

            payload = response.json()
            if is_crypto:
                if not isinstance(payload, list):
                    raise ValueError("Tiingo crypto payload format is not a list.")
                return _normalize_tiingo_crypto_payload(payload, ticker=ticker)
            if not isinstance(payload, list):
                raise ValueError("Tiingo payload format is not a list.")
            return _normalize_tiingo_payload(payload, ticker=ticker)

        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt) + random.uniform(1.0, 3.0)
            LOGGER.warning(
                "Tiingo request failed | ticker=%s | attempt=%d/%d | error=%s",
                ticker,
                attempt_no,
                max_retries,
                exc,
            )
            if attempt_no < max_retries:
                LOGGER.info("Retrying Tiingo in %.2f seconds.", delay)
                time.sleep(delay)

    if last_exc is None:
        raise RuntimeError("Tiingo request failed with unknown error.")
    raise RuntimeError(f"Tiingo request failed after {max_retries} attempts: {last_exc}")
