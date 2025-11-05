import numpy as np
import pandas as pd

def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # tenta achar coluna 'close' (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    if 'close' not in cols:
        raise ValueError("CSV deve conter coluna 'close' (case-insensitive).")
    # ordena por tempo se existir 'timestamp'/'date'
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

def make_windows(r: np.ndarray, L: int, step: int, horizon: int):
    """Cria janelas de retornos e labels binários para horizonte H.
    Label = 1{sum_{t+1..t+H} r > 0}
    Retorna: X_raw (N,L), y (N,), idxs (índices de início)
    """
    X, y, idxs = [], [], []
    T = len(r)
    for start in range(0, T - L - horizon, step):
        end = start + L
        future = r[end:end+horizon].sum()
        X.append(r[start:end])
        y.append(1 if future > 0 else 0)
        idxs.append(start)
    return np.array(X), np.array(y), np.array(idxs)

def zscore_per_asset(r: np.ndarray, eps: float=1e-8):
    mu, sd = r.mean(), r.std()
    sd = sd if sd > eps else eps
    return (r - mu) / sd

def feature_block(xw: np.ndarray) -> np.ndarray:
    """Extrai features simples por janela (batch, L) -> (batch, D)."""
    import numpy as np
    from scipy.stats import skew, kurtosis
    B, L = xw.shape
    feats = []
    for i in range(B):
        x = xw[i]
        m = x.mean(); s = x.std() + 1e-8
        sk = skew(x); kt = kurtosis(x, fisher=True)  # 0=gauss
        mom = x[-10:].sum() if L >= 10 else x.sum()
        # autocorr simples de r^2 (lag 1)
        x2 = (x**2) - (x**2).mean()
        denom = (x2**2).sum() + 1e-12
        ac_r2 = (x2[:-1]*x2[1:]).sum()/denom if L>1 else 0.0
        feats.append([m, s, sk, kt, mom, ac_r2])
    return np.array(feats)
