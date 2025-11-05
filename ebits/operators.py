import numpy as np
from typing import Tuple

def _bounded(x, lo, hi): return np.clip(x, lo, hi)

def op_local_impulse(r: np.ndarray, cfg, strength=1.0) -> np.ndarray:
    out = r.copy()
    idx = np.random.randint(0, len(r))
    amp = np.random.randn() * strength
    out[idx] += amp
    return out

def op_trend_bend(r: np.ndarray, cfg, strength=0.5) -> np.ndarray:
    out = r.copy()
    slope = np.random.randn() * (strength / max(len(r),1))
    trend = slope * (np.arange(len(r)) - len(r)/2)
    return out + trend

def op_vol_burst(r: np.ndarray, cfg, strength=1.0) -> np.ndarray:
    out = r.copy()
    L = max(1, np.random.randint(max(1, len(r)//16), max(2, len(r)//6)))
    s = np.random.randint(0, len(r)-L+1)
    scale = 1.0 + abs(np.random.randn()) * strength
    out[s:s+L] *= scale
    out = _bounded(out, -cfg.max_abs_jump, cfg.max_abs_jump)
    return out

def op_warp_time(r: np.ndarray, cfg, strength=0.2) -> np.ndarray:
    n = len(r)
    if n < 3: return r.copy()
    k = max(3, n//64)
    knots = np.linspace(0, n-1, k).astype(int)
    jitter = (np.random.randn(k) * strength * (n/k)).astype(int)
    tgt = np.clip(knots + jitter, 0, n-1)
    tgt[0], tgt[-1] = 0, n-1
    return np.interp(np.arange(n), tgt, r[knots])

def crossover_one_point(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(a)
    if n < 2: return a.copy(), b.copy()
    c = np.random.randint(1, n-1)
    return np.concatenate([a[:c], b[c:]]), np.concatenate([b[:c], a[c:]])

def mutate(r: np.ndarray, cfg) -> np.ndarray:
    ops = [op_local_impulse, op_trend_bend, op_vol_burst, op_warp_time]
    out = r.copy()
    n_ops = np.random.randint(1, 3)
    for _ in range(n_ops):
        op = np.random.choice(ops)
        out = op(out, cfg)
    return out
