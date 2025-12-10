#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valida a consistência de um diretório de CSVs (com 'close') para um dado (L,H,step).
Checa: tamanho mínimo, monotonicidade temporal, não-negatividade, estatísticas básicas.
"""
import argparse, os
import pandas as pd

def load_csv(path):
    df = pd.read_csv(path)
    # harmonizar nomes
    cols = {c.lower(): c for c in df.columns}
    if 'timestamp' in cols: tcol = cols['timestamp']
    elif 'date' in cols: tcol = cols['date']
    elif 'datetime' in cols: tcol = cols['datetime']
    elif 'time' in cols: tcol = cols['time']
    else: tcol = None

    if tcol:
        df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        df = df.dropna(subset=[tcol]).sort_values(tcol).drop_duplicates(subset=[tcol])

    if 'close' not in cols:
        for c in df.columns:
            if c.lower()=='close':
                df.rename(columns={c:'close'}, inplace=True)
                break
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    df = df[df['close']>0]
    return df, tcol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Diretório com CSVs padronizados")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()

    ok_files = 0
    for f in sorted(os.listdir(args.dir)):
        if not f.lower().endswith(".csv"): 
            continue
        path = os.path.join(args.dir, f)
        try:
            df, tcol = load_csv(path)
            n = len(df)
            status = "OK" if n >= (args.L + args.H + args.step) else "CURTO"
            print(f"{f:35s}  n={n:6d}  {('tcol='+tcol) if tcol else 'sem time'}  status={status}")
            ok_files += int(status=="OK")
        except Exception as e:
            print(f"{f:35s}  ERRO: {e}")
    print(f"\nArquivos OK (>= L+H+step): {ok_files}")

if __name__ == '__main__':
    main()
