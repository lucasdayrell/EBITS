import os, glob
import numpy as np
import pandas as pd

def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if 'close' not in cols:
        raise ValueError("CSV deve conter coluna 'close' (case-insensitive).")
    for tcol in ['timestamp','date','datetime','time']:
        if tcol in cols:
            df[cols[tcol]] = pd.to_datetime(df[cols[tcol]])
            df = df.sort_values(cols[tcol])
            break
    return df

def compute_log_returns(close: pd.Series) -> np.ndarray:
    close = close.astype(float).replace(0, np.nan).ffill()
    r = np.log(close).diff().dropna().values
    return r

def zscore_per_asset(r: np.ndarray, eps: float=1e-8):
    mu, sd = r.mean(), r.std()
    sd = sd if sd > eps else eps
    return (r - mu) / sd

def make_windows(r: np.ndarray, L: int, step: int, horizon: int):
    X, y, idxs = [], [], []
    T = len(r)
    for start in range(0, T - L - horizon, step):
        end = start + L
        future = r[end:end+horizon].sum()
        X.append(r[start:end])
        y.append(1 if future > 0 else 0)
        idxs.append(start)
    return np.array(X), np.array(y), np.array(idxs)

def feature_block(xw: np.ndarray) -> np.ndarray:
    import numpy as np
    from scipy.stats import skew, kurtosis
    B, L = xw.shape
    feats = []
    for i in range(B):
        x = xw[i]
        m = x.mean(); s = x.std() + 1e-8
        sk = skew(x); kt = kurtosis(x, fisher=True)
        mom = x[-10:].sum() if L >= 10 else x.sum()
        x2 = (x**2) - (x**2).mean()
        denom = (x2**2).sum() + 1e-12
        ac_r2 = (x2[:-1]*x2[1:]).sum()/denom if L>1 else 0.0
        feats.append([m, s, sk, kt, mom, ac_r2])
    return np.array(feats)

def build_window_bank(csv_dir: str, L: int, step: int, horizon: int, limit_files: int = 200):
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))[:limit_files]
    bank = []
    for fp in files:
        try:
            df = load_ohlcv_csv(fp)
            close = df[[c for c in df.columns if c.lower()=='close'][0]]
            r = compute_log_returns(close)
            r = zscore_per_asset(r)
            X, _, _ = make_windows(r, L=L, step=step, horizon=horizon)
            if len(X):
                bank.append(X)
        except Exception:
            continue
    if not bank:
        raise ValueError("Nenhuma janela construída. Verifique CSVs.")
    return np.vstack(bank)

def initial_population_from_bank(r0: np.ndarray, bank: np.ndarray, pop_size: int, noise_sigma: float = 0.01):
    diffs = bank - r0[None, :]
    l2 = np.linalg.norm(diffs, axis=1)
    K = min(max(pop_size*5, pop_size), len(bank))
    idxs = np.argsort(l2)[:K]
    base = bank[idxs]
    sel = base[np.random.randint(0, len(base), size=pop_size)]
    noise = np.random.randn(pop_size, r0.shape[0]) * noise_sigma
    return sel + noise