#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Price Analyzer
Fetches historical prices for a list of cryptocurrencies for the last N days
(default 1000) from CoinGecko and outputs:
- A tidy CSV and an Excel workbook with per-coin OHLCV-like columns (price only from CoinGecko + derived returns)
- Per-coin PNG charts (price, log-price, and daily returns)
- A normalized comparison chart (all coins = 100 at start)
- A summary table with key stats (CAGR, max drawdown, volatility, Sharpe-like ratio with rf=0)

Usage examples:
  python crypto_price_analyzer.py --coins BTC,ETH,BNB,SOL,HYPE,XRP,DOGE,TRX,ADA,LINK --days 1000
  python crypto_price_analyzer.py --config config.json

Notes:
- Data source: CoinGecko public API (no key). Endpoint: /coins/{id}/market_chart?vs_currency=usd&days={days}
- Coin IDs differ from tickers. We map tickers -> IDs automatically when possible, otherwise we try to disambiguate.
- "HYPE" may be ambiguous on CoinGecko. If it can’t be uniquely resolved, it will be skipped with a warning.
"""
import argparse
import sys
import time
from typing import List, Dict, Tuple, Optional
import math
import os
import json
import datetime as dt

try:
    import requests
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print("Please install dependencies first: pip install requests pandas numpy matplotlib", file=sys.stderr)
    raise

COINGECKO_API = "https://api.coingecko.com/api/v3"
DEFAULT_COINS = "BTC,ETH,BNB,SOL,HYPE,XRP,DOGE,TRX,ADA,LINK"
DEFAULT_DAYS = 1000
VS_CCY = "usd"

# Known mappings to speed things up and avoid ambiguity.
# Feel free to extend this list.
KNOWN_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "TRX": "tron",
    "ADA": "cardano",
    "LINK": "chainlink",
    # HYPE intentionally omitted: we will resolve dynamically
}

def _get_all_coins_list() -> pd.DataFrame:
    """Download CoinGecko coins list and return as DataFrame (id, symbol, name)."""
    url = f"{COINGECKO_API}/coins/list?include_platform=false"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    # Normalize case for symbols for matching convenience.
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].str.upper()
    return df[["id", "symbol", "name"]]

def _resolve_symbol_to_id(symbols: List[str]) -> Dict[str, Optional[str]]:
    """Resolve list of ticker symbols to CoinGecko IDs.
    Returns a dict symbol->id (or None if unresolved/ambiguous)."""
    coins_df = _get_all_coins_list()
    out = {}
    for sym in symbols:
        s = sym.upper().strip()
        if s in KNOWN_MAP:
            out[s] = KNOWN_MAP[s]
            continue
        candidates = coins_df[coins_df["symbol"] == s]
        if len(candidates) == 1:
            out[s] = candidates.iloc[0]["id"]
        elif len(candidates) > 1:
            # try exact name match heuristic (rarely helpful for tickers, but harmless)
            name_candidates = candidates[candidates["name"].str.upper() == s]
            if len(name_candidates) == 1:
                out[s] = name_candidates.iloc[0]["id"]
            else:
                # ambiguous, pick the most popular if we can (requires extra API) otherwise None
                out[s] = None
        else:
            out[s] = None
    return out

def _fetch_market_chart(coin_id: str, days: int, vs_currency: str = VS_CCY) -> pd.DataFrame:
    """Fetch market chart (prices) for coin_id over 'days' lookback. Returns DataFrame with datetime and price."""
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=60)
    if r.status_code == 429:
        # rate-limited, backoff once
        time.sleep(2)
        r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    prices = j.get("prices", [])
    if not prices:
        raise ValueError(f"No price data returned for {coin_id}")
    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("Europe/Riga").dt.date
    df = df.drop(columns=["ts_ms"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def _compute_metrics(price_series: pd.Series) -> Dict[str, float]:
    """Compute simple metrics from a daily price series."""
    s = price_series.dropna().astype(float)
    if len(s) < 2:
        return {"cagr": np.nan, "vol": np.nan, "max_dd": np.nan, "sharpe0": np.nan}
    # Daily returns
    ret = s.pct_change()
    # CAGR (approx): (End/Start)^(365/n)-1
    n_days = len(s)
    try:
        cagr = (s.iloc[-1] / s.iloc[0]) ** (365.0 / n_days) - 1.0
    except Exception:
        cagr = np.nan
    vol = ret.std() * math.sqrt(365.0)
    # Max drawdown
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    max_dd = dd.min()
    sharpe0 = (ret.mean() * 365.0) / (ret.std() * math.sqrt(365.0)) if ret.std() > 0 else np.nan
    return {"cagr": cagr, "vol": vol, "max_dd": max_dd, "sharpe0": sharpe0}

def analyze(coins: List[str], days: int, out_dir: str = "output"):
    os.makedirs(out_dir, exist_ok=True)

    # Resolve tickers -> ids
    mapping = _resolve_symbol_to_id(coins)
    unresolved = [k for k, v in mapping.items() if v is None]
    if unresolved:
        print(f"[WARN] Could not uniquely resolve tickers: {', '.join(unresolved)}. They will be skipped.", file=sys.stderr)

    series_frames = []
    metrics_rows = []
    normalized = pd.DataFrame()

    for sym in coins:
        cid = mapping.get(sym.upper())
        if not cid:
            continue
        try:
            df = _fetch_market_chart(cid, days, VS_CCY)
            df.rename(columns={"price": f"{sym.upper()}_price"}, inplace=True)
            # Merge into a wide table by date
            if len(series_frames) == 0:
                wide = df.copy()
            else:
                wide = wide.merge(df, on="date", how="outer")
            series_frames.append(df)

        except Exception as e:
            print(f"[ERR] Failed to fetch {sym} ({cid}): {e}", file=sys.stderr)

    if not series_frames:
        raise SystemExit("No data downloaded. Exiting.")

    wide = wide.sort_values("date").reset_index(drop=True)
    # create tidy/long table for convenience
    long_rows = []
    for col in wide.columns:
        if col.endswith("_price"):
            sym = col.replace("_price", "")
            sub = wide[["date", col]].rename(columns={col: "price"}).copy()
            sub["symbol"] = sym
            # Daily return
            sub["return"] = sub["price"].pct_change()
            long_rows.append(sub)
    tidy = pd.concat(long_rows, ignore_index=True)

    # Metrics
    for sym in sorted(set(tidy["symbol"])):
        s = tidy.loc[tidy["symbol"] == sym, "price"]
        m = _compute_metrics(s)
        metrics_rows.append({"symbol": sym, **m})
    metrics = pd.DataFrame(metrics_rows).set_index("symbol").sort_index()

    # Normalized index = 100 at start (per coin)
    norm = {}
    for sym in sorted(set(tidy["symbol"])):
        s = tidy.loc[tidy["symbol"] == sym, ["date", "price"]].dropna()
        if s.empty:
            continue
        base = float(s["price"].iloc[0])
        norm[sym] = (s["price"] / base) * 100.0
    normalized = pd.concat(norm, axis=1)
    normalized.index = tidy.loc[tidy["symbol"] == sorted(set(tidy["symbol"]))[0], "date"].values[: len(normalized)] if not normalized.empty else []

    # Save tables
    csv_path = os.path.join(out_dir, "prices_tidy.csv")
    tidy.to_csv(csv_path, index=False, encoding="utf-8")
    xlsx_path = os.path.join(out_dir, "prices_and_metrics.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xl:
        wide.to_excel(xl, sheet_name="Wide_Prices", index=False)
        tidy.to_excel(xl, sheet_name="Tidy_Prices", index=False)
        metrics.to_excel(xl, sheet_name="Metrics")
        if not normalized.empty:
            normalized.to_excel(xl, sheet_name="Normalized_100")

    # Plotting
    plt.ioff()

    # Per-coin charts
    for sym in sorted(set(tidy["symbol"])):
        sub = tidy[tidy["symbol"] == sym].dropna(subset=["price"]).copy()
        if sub.empty:
            continue

        # Price
        fig = plt.figure()
        plt.plot(sub["date"], sub["price"])
        plt.title(f"{sym} Price (USD) – last {days} days")
        plt.xlabel("Date"); plt.ylabel("Price (USD)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{sym}_price.png"))
        plt.close(fig)

        # Log price
        fig = plt.figure()
        plt.plot(sub["date"], np.log(sub["price"]))
        plt.title(f"{sym} Log-Price – last {days} days")
        plt.xlabel("Date"); plt.ylabel("log(Price)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{sym}_log_price.png"))
        plt.close(fig)

        # Daily returns
        fig = plt.figure()
        plt.plot(sub["date"], sub["return"].fillna(0.0))
        plt.title(f"{sym} Daily Returns – last {days} days")
        plt.xlabel("Date"); plt.ylabel("Return")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{sym}_returns.png"))
        plt.close(fig)

    # Normalized comparison
    if not normalized.empty:
        fig = plt.figure()
        for col in normalized.columns:
            plt.plot(normalized.index, normalized[col], label=col)
        plt.title(f"Normalized Performance (start=100) – last {days} days")
        plt.xlabel("Date"); plt.ylabel("Index (start=100)")
        plt.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "normalized_comparison.png"))
        plt.close(fig)

    # Save mapping and summary
    with open(os.path.join(out_dir, "resolved_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    metrics_path = os.path.join(out_dir, "metrics_summary.csv")
    metrics.to_csv(metrics_path, encoding="utf-8")

    print(f"Done.\nOutputs saved to: {out_dir}\n  - {csv_path}\n  - {xlsx_path}\n  - charts PNGs (per coin + normalized)\n  - resolved_mapping.json\n  - metrics_summary.csv")

def parse_args():
    p = argparse.ArgumentParser(description="Analyze crypto price movements from CoinGecko")
    p.add_argument("--coins", type=str, default=DEFAULT_COINS,
                   help="Comma-separated tickers (e.g., BTC,ETH,BNB,SOL,HYPE,XRP,DOGE,TRX,ADA,LINK)")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="How many days back (e.g., 1000, 365, 'max' not supported here)")
    p.add_argument("--out", type=str, default="output", help="Output directory")
    p.add_argument("--config", type=str, default=None, help="Optional JSON config path overriding args")
    args = p.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        coins = cfg.get("coins", DEFAULT_COINS)
        days = int(cfg.get("days", DEFAULT_DAYS))
        out_dir = cfg.get("out", "output")
    else:
        coins = args.coins
        days = int(args.days)
        out_dir = args.out

    coins_list = [c.strip() for c in coins.split(",") if c.strip()]
    return coins_list, days, out_dir

if __name__ == "__main__":
    coins_list, days, out_dir = parse_args()
    analyze(coins_list, days, out_dir)
