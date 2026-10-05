#!/usr/bin/env python3
"""
crypto-bull-regime-system · descarga de datos

Descarga velas DIARIAS (OHLCV, en USDT) de la API pública de Binance, sin clave,
para el universo del sistema. Cada moneda se guarda en data/<SÍMBOLO>.csv.
Las velas diarias de Binance cierran a las 00:00 UTC, igual que los datos del 006.

Uso:
    python fetch_data.py
    python fetch_data.py --symbols BTC ETH SOL
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

UNIVERSE = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "LINK", "AVAX"]
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
HEADERS = {"User-Agent": "crypto-bull-regime-system/1.0 (research)"}
START_MS = int(pd.Timestamp("2017-01-01").timestamp() * 1000)


def _get(url: str) -> list:
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (400, 403, 404, 451):
                raise
            time.sleep(2 * (i + 1))
        except urllib.error.URLError:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"No se pudo descargar {url}")


def fetch_symbol(sym: str) -> pd.DataFrame:
    pair = f"{sym}USDT"
    last_err = None
    for host in HOSTS:
        try:
            rows, start = [], START_MS
            while True:
                q = urllib.parse.urlencode({"symbol": pair, "interval": "1d",
                                            "startTime": start, "limit": 1000})
                batch = _get(f"{host}/api/v3/klines?{q}")
                if not batch:
                    break
                rows.extend(batch)
                start = batch[-1][0] + 86_400_000
                if len(batch) < 1000:
                    break
                time.sleep(0.2)
            df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume",
                                             "close_time", "quote_volume", "trades", "tb_base",
                                             "tb_quote", "ignore"])
            df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
            df = df[["date", "open", "high", "low", "close", "volume", "quote_volume"]]
            for c in df.columns[1:]:
                df[c] = pd.to_numeric(df[c])
            # descarta la vela del día en curso (aún no cerrada)
            df = df[df["date"] < pd.Timestamp.utcnow().tz_localize(None).normalize()]
            return df.drop_duplicates("date").set_index("date")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"{pair}: {last_err}")


def main():
    ap = argparse.ArgumentParser(description="Descarga velas diarias de Binance")
    ap.add_argument("--symbols", nargs="+", default=UNIVERSE)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for sym in a.symbols:
        try:
            df = fetch_symbol(sym)
            df.to_csv(os.path.join(a.out, f"{sym}.csv"))
            print(f"  OK  {sym:5s} {len(df):>5,} días  ({df.index[0].date()} -> {df.index[-1].date()})")
        except Exception as e:
            print(f"  --  {sym:5s} error: {e}")


if __name__ == "__main__":
    main()
