#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetcher de datasets para o EBITS:
- Binance Klines (intraday)  -> OHLCV de cripto (1m/5m/etc), timestamp em UTC
- Stooq (diário)             -> OHLCV de ações/FX/índices (CSV direto)
- FRED (macro)               -> séries econômicas (mensal/semanal/diária), via API

Saída: CSVs padronizados com pelo menos colunas ['timestamp','close'], ordenados em ordem crescente de tempo,
sem duplicatas, com validações básicas para janelas consistentes.
"""
import argparse, os, sys, time, io, json, datetime as dt
from typing import Optional
import pandas as pd
import numpy as np
import requests


# ---------------------- Utils ----------------------
def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _to_ms(date_str: str) -> int:
    """YYYY-MM-DD -> epoch ms (UTC 00:00:00)"""
    d = dt.datetime.strptime(date_str, "%Y-%m-%d")
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

def _to_utc_iso(ts_ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ts_ms/1000.0).strftime("%Y-%m-%d %H:%M:%S")

def _sanitize_numeric_close(df: pd.DataFrame) -> pd.DataFrame:
    """Garante coluna 'close' numérica e positiva (remove NaN e <=0)."""
    if "close" not in df.columns:
        for c in df.columns:
            if c.lower() == "close":
                df.rename(columns={c: "close"}, inplace=True)
                break
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df

def _finalize_timeseries(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Ordena, deduplica por timestamp e mantém monotonicidade temporal."""
    df = df.dropna(subset=[time_col]).copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col).drop_duplicates(subset=[time_col])
    return df

def _save_csv(df: pd.DataFrame, path: str):
    if "close" not in df.columns:
        raise ValueError("DataFrame sem coluna 'close'.")
    df.to_csv(path, index=False)
    print(f"[ok] salvo: {path} (n={len(df)})")


# ---------------------- Binance (intraday) ----------------------
def fetch_binance_klines(symbol: str, interval: str, start: str, end: str, throttle_s: float=0.2) -> pd.DataFrame:
    """
    Pagina klines da Binance (spot) de start..end (YYYY-MM-DD) no intervalo especificado.
    Retorna DataFrame com ['timestamp','open','high','low','close','volume'] em UTC.
    """
    base = "https://api.binance.com/api/v3/klines"
    start_ms = _to_ms(start)
    end_ms   = _to_ms(end) + 24*3600*1000 - 1
    out = []
    params = {"symbol": symbol.upper(), "interval": interval, "limit": 1000, "startTime": start_ms}
    while True:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data: 
            break
        for k in data:
            # kline: [openTime, open, high, low, close, volume, closeTime, ...]
            open_t  = int(k[0])
            close_t = int(k[6])
            if open_t > end_ms:
                break
            out.append({
                "timestamp": _to_utc_iso(open_t),
                "open": float(k[1]),
                "high": float(k[2]),
                "low":  float(k[3]),
                "close":float(k[4]),
                "volume": float(k[5]),
                "close_time": _to_utc_iso(close_t)
            })
        if len(data) < 1000:
            break
        params["startTime"] = int(data[-1][6]) + 1  # se move pelo closeTime
        if params["startTime"] > end_ms:
            break
        time.sleep(throttle_s)
    if not out:
        raise RuntimeError(f"Nenhuma barra para {symbol} {interval} em {start}..{end}")
    df = pd.DataFrame(out)
    df = _sanitize_numeric_close(df)
    df = _finalize_timeseries(df, "timestamp")
    return df[["timestamp","open","high","low","close","volume"]]


# ---------------------- Stooq (diário) ----------------------
def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    """
    Baixa CSV diário da Stooq. O ticker precisa estar no formato que a Stooq reconhece
    (ex.: 'spy.us', 'qqq.us', 'eurusd'). Se você passar 'SPY', tentamos variantes comuns.
    """
    base = "https://stooq.com/q/d/l/"
    tried = []

    def _try(sym: str):
        url = f"{base}?s={sym}&i=d"
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or not r.text or r.text.lower().startswith("error"):
            return None
        df = pd.read_csv(io.StringIO(r.text))
        if "Close" not in df.columns:
            return None
        df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        df = _sanitize_numeric_close(df)
        df = _finalize_timeseries(df, "timestamp")
        return df[["timestamp","open","high","low","close","volume"]]

    cand = [ticker, ticker.lower()]
    sym = ticker.lower()
    if "." not in sym:  # tentar sufixos usuais
        cand += [f"{sym}.us", f"{sym}.jp", f"{sym}.de", f"{sym}.pl", f"{sym}.uk", f"{sym}.br"]

    for c in cand:
        tried.append(c)
        df = _try(c)
        if df is not None and len(df):
            return df
    raise RuntimeError(f"Stooq não baixou '{ticker}' (tentativas: {tried})")


# ---------------------- FRED (macro) ----------------------
def fetch_fred_series(series_id: str, api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Baixa observações da série FRED (qualquer frequência). Gera CSV com ['timestamp','close'].
    """
    base = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end
    }
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    obs = js.get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED sem dados: {series_id} ({start}..{end})")
    rows = []
    for o in obs:
        v = o.get("value", None)
        if v in (None, ".", ""):  # FRED usa "." para missing
            continue
        rows.append({"timestamp": o["date"], "close": float(v)})
    df = pd.DataFrame(rows)
    df = _sanitize_numeric_close(df)
    df = _finalize_timeseries(df, "timestamp")
    return df[["timestamp","close"]]


# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(description="Fetcher de datasets (Binance/Stooq/FRED) para EBITS")
    # Pastas destino
    ap.add_argument("--out-intraday-dir", type=str, default="data/intraday", help="Pasta de saída para intraday (Binance)")
    ap.add_argument("--out-daily-dir", type=str, default="data/daily", help="Pasta de saída para diário (Stooq)")
    ap.add_argument("--out-macro-dir", type=str, default="data/daily_macro", help="Pasta de saída para macro (FRED)")
    # Binance
    ap.add_argument("--binance-symbols", type=str, default="BTCUSDT,ETHUSDT", help="Símbolos separados por vírgula (ex.: BTCUSDT,ETHUSDT)")
    ap.add_argument("--binance-interval", type=str, default="1m", help="Intervalo (ex.: 1m,5m,15m,1h)")
    ap.add_argument("--binance-start", type=str, default="2023-01-01")
    ap.add_argument("--binance-end", type=str, default=dt.datetime.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--binance-throttle", type=float, default=0.2, help="Sleep (s) entre páginas para evitar rate limit")
    # Stooq
    ap.add_argument("--stooq-tickers", type=str, default="spy.us,qqq.us,eurusd", help="Tickers Stooq (ex.: spy.us,qqq.us,eurusd)")
    # FRED
    ap.add_argument("--fred-series", type=str, default="CPIAUCSL,UNRATE,DGS10", help="IDs FRED (ex.: CPIAUCSL,UNRATE,DGS10)")
    ap.add_argument("--fred-key", type=str, default=os.environ.get("FRED_API_KEY",""), help="API key FRED (ou defina env FRED_API_KEY)")
    ap.add_argument("--fred-start", type=str, default="1990-01-01")
    ap.add_argument("--fred-end", type=str, default=dt.datetime.utcnow().strftime("%Y-%m-%d"))

    args = ap.parse_args()

    # Pastas
    _ensure_dir(args.out_intraday_dir)
    _ensure_dir(args.out_daily_dir)
    _ensure_dir(args.out_macro_dir)

    # -------- Binance --------
    if args.binance_symbols:
        for sym in [s.strip() for s in args.binance_symbols.split(",") if s.strip()]:
            print(f"[Binance] {sym} {args.binance_interval} {args.binance_start}..{args.binance_end}")
            try:
                df = fetch_binance_klines(sym, args.binance_interval, args.binance_start, args.binance_end, args.binance_throttle)
                out = os.path.join(args.out_intraday_dir, f"{sym}_{args.binance_interval}.csv")
                _save_csv(df, out)
                print(f"  período: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}  n={len(df)}")
            except Exception as e:
                print(f"[ERRO][Binance] {sym}: {e}")

    # -------- Stooq --------
    if args.stooq_tickers:
        for tic in [t.strip() for t in args.stooq_tickers.split(",") if t.strip()]:
            print(f"[Stooq] {tic}")
            try:
                df = fetch_stooq_daily(tic)
                out = os.path.join(args.out_daily_dir, f"{tic.replace('.','_')}.csv")
                _save_csv(df, out)
                print(f"  período: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}  n={len(df)}")
            except Exception as e:
                print(f"[ERRO][Stooq] {tic}: {e}")

    # -------- FRED --------
    if args.fred_series and args.fred_key:
        for sid in [s.strip() for s in args.fred_series.split(",") if s.strip()]:
            print(f"[FRED] {sid} {args.fred_start}..{args.fred_end}")
            try:
                df = fetch_fred_series(sid, args.fred_key, args.fred_start, args.fred_end)
                out = os.path.join(args.out_macro_dir, f"{sid}.csv")
                _save_csv(df, out)
                print(f"  período: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}  n={len(df)}")
            except Exception as e:
                print(f"[ERRO][FRED] {sid}: {e}")
    elif args.fred_series and not args.fred_key:
        print("[AVISO] FRED: sem --fred-key / FRED_API_KEY; pulando FRED.")

    print("\nTudo pronto. Use as pastas como --bank_intraday e --bank_daily no run_experiment.py.")
    print("Dica: mantenha intraday e diário separados para janelas e horizontes coerentes.")

if __name__ == "__main__":
    main()
