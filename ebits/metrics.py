import numpy as np
from math import sqrt
try:
    from fastdtw import fastdtw
    _HAS_FASTDTW = True
except Exception:
    _HAS_FASTDTW = False

def l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))

def dtw_distance(x: np.ndarray, y: np.ndarray, radius: int = 1) -> float:
    if _HAS_FASTDTW:
        d, _ = fastdtw(x, y, radius=radius)
        return float(sqrt(d))
    n, m = len(x), len(y)
    D = np.full((n+1, m+1), np.inf); D[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = (x[i-1] - y[j-1])**2
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(np.sqrt(D[n,m]))

def acf(x: np.ndarray, max_lag: int=20) -> np.ndarray:
    x = x - x.mean()
    denom = (x**2).sum() + 1e-12
    return np.array([np.dot(x[:-k], x[k:]) / denom if k>0 else 1.0 for k in range(max_lag+1)])

def skew_kurt(x: np.ndarray):
    from scipy.stats import skew, kurtosis
    return float(skew(x)), float(kurtosis(x, fisher=True))

def jarque_bera(x: np.ndarray):
    n = len(x)
    if n < 3: return 0.0
    s, k = skew_kurt(x)
    return float(n/6.0 * (s*s + 0.25*(k*k)))

def spectral_energy(x: np.ndarray, nbands: int = 6):
    xf = np.fft.rfft(x - x.mean())
    ps = (xf.real**2 + xf.imag**2)
    ps = ps / (ps.sum() + 1e-12)
    splits = np.array_split(ps, nbands)
    return np.array([s.sum() for s in splits])

def plausibility_penalty(r: np.ndarray, r0: np.ndarray, cfg) -> float:
    acf_r, acf_r0 = acf(r, cfg.acf_max_lag), acf(r0, cfg.acf_max_lag)
    acf_r2, acf_r20 = acf(r**2, cfg.acf_max_lag), acf(r0**2, cfg.acf_max_lag)
    d_acf  = np.linalg.norm(acf_r - acf_r0)
    d_acf2 = np.linalg.norm(acf_r2 - acf_r20)
    s, k = skew_kurt(r); s0, k0 = skew_kurt(r0)
    d_sk = abs(s - s0) + abs(k - k0)
    jb_r, jb_r0 = jarque_bera(r), jarque_bera(r0)
    d_jb = abs(jb_r - jb_r0)
    e, e0 = spectral_energy(r), spectral_energy(r0)
    d_spec = np.linalg.norm(e - e0)
    pen = (cfg.w_acf  * d_acf +
           cfg.w_acf2 * d_acf2 +
           cfg.w_skewkurt * d_sk +
           cfg.w_jb * d_jb +
           cfg.w_spectrum * d_spec)
    return float(pen)

def compute_summary_metrics(model, target_class: int, r0_batch, results):
    flips = 0; n = len(results)
    delta_confs = []; dtws = []; l2s = []; plaus = []
    for res in results:
        r0 = res["r0"]
        P0 = float(model.predict_proba(r0[None, :, None])[0, target_class])
        objs = res["objs"]; pop = res["pop"]
        best_idx = int(np.argmin(objs[:,0]))
        r_star = pop[best_idx]
        P1 = float(model.predict_proba(r_star[None, :, None])[0, target_class])
        delta_confs.append(P1 - P0)
        y0 = int(np.argmax(model.predict_proba(r0[None, :, None])[0]))
        y1 = int(np.argmax(model.predict_proba(r_star[None, :, None])[0]))
        if y0 != y1: flips += 1
        dtws.append(objs[best_idx][1])
        l2s.append(float(np.linalg.norm(r_star - r0)))
        plaus.append(objs[best_idx][2])
    metrics = {
        "flip_rate": flips / max(n,1),
        "mean_delta_conf": float(np.mean(delta_confs)) if delta_confs else 0.0,
        "mean_dtw": float(np.mean(dtws)) if dtws else 0.0,
        "mean_l2": float(np.mean(l2s)) if l2s else 0.0,
        "mean_plaus": float(np.mean(plaus)) if plaus else 0.0
    }
    return metrics