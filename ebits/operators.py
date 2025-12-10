import numpy as np

def _bounded(x, lo, hi): return np.clip(x, lo, hi)

def op_local_impulse(r, cfg, strength=1.0):
    out = r.copy()
    if len(r)==0: return out
    idx = np.random.randint(0, len(r))
    amp = np.random.randn() * strength
    out[idx] += amp
    return _bounded(out, -cfg.max_abs_jump, cfg.max_abs_jump)

def op_trend_bend(r, cfg, strength=0.5):
    out = r.copy()
    n = max(len(r),1)
    slope = np.random.randn() * (strength / n)
    trend = slope * (np.arange(n) - n/2)
    out = out + trend
    return _bounded(out, -cfg.max_abs_jump, cfg.max_abs_jump)

def op_vol_burst(r, cfg, strength=1.0):
    out = r.copy()
    n = max(len(r),1)
    L = max(1, np.random.randint(max(1, n//16), max(2, n//6)))
    s = np.random.randint(0, n-L+1)
    scale = 1.0 + abs(np.random.randn()) * strength
    out[s:s+L] *= scale
    return _bounded(out, -cfg.max_abs_jump, cfg.max_abs_jump)

def op_warp_time(r, cfg, strength=0.2):
    n = len(r)
    if n < 3: return r.copy()
    k = max(3, n//64)
    knots = np.linspace(0, n-1, k).astype(int)
    jitter = (np.random.randn(k) * strength * (n/k)).astype(int)
    tgt = np.clip(knots + jitter, 0, n-1)
    tgt[0], tgt[-1] = 0, n-1
    warped = np.interp(np.arange(n), tgt, r[knots])
    return _bounded(warped, -cfg.max_abs_jump, cfg.max_abs_jump)

def crossover_one_point(a, b):
    n = len(a)
    if n < 2: return a.copy(), b.copy()
    c = np.random.randint(1, n-1)
    return np.concatenate([a[:c], b[c:]]), np.concatenate([b[:c], a[c:]])

def mutate(r, cfg):
    ops = [
        ("impulse", op_local_impulse),
        ("trend",   op_trend_bend),
        ("vol",     op_vol_burst),
        ("warp",    op_warp_time),
    ]
    names, funs = zip(*ops)
    probs = np.array([cfg.op_probs.get(n, 0.0) for n in names], dtype=float)
    probs = probs / probs.sum() if probs.sum()>0 else np.ones_like(probs)/len(probs)
    out = r.copy()
    n_ops = np.random.randint(1, 3)
    for _ in range(n_ops):
        j = np.random.choice(len(funs), p=probs)
        strength = float(cfg.op_strength.get(names[j], 1.0))
        out = funs[j](out, cfg, strength=strength)
    return out