import numpy as np

def l2_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))

def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
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

def plausibility_penalty(r: np.ndarray, r0: np.ndarray, max_lag: int=20) -> float:
    acf_r, acf_r0 = acf(r, max_lag), acf(r0, max_lag)
    acf_r2, acf_r20 = acf(r**2, max_lag), acf(r0**2, max_lag)
    d1 = np.linalg.norm(acf_r - acf_r0)
    d2 = np.linalg.norm(acf_r2 - acf_r20)
    return float(d1 + d2)

def compute_summary_metrics(model, target_class: int, r0_batch, results):
    """Agrega flip rate, delta conf, proximidade e plausibilidade médios."""
    flips = 0; n = len(results)
    delta_confs = []; dtws = []; l2s = []; plaus = []
    for b, res in enumerate(results):
        r0 = res["r0"]
        P0 = float(model.predict_proba(r0[None, :, None])[0, target_class])
        # escolhe candidato com maior prob alvo
        objs = res["objs"]; pop = res["pop"]
        best_idx = int(np.argmin(objs[:,0]))
        r_star = pop[best_idx]
        P1 = float(model.predict_proba(r_star[None, :, None])[0, target_class])
        delta_confs.append(P1 - P0)
        # flip check
        y0 = int(np.argmax(model.predict_proba(r0[None, :, None])[0]))
        y1 = int(np.argmax(model.predict_proba(r_star[None, :, None])[0]))
        if y0 != y1: flips += 1
        # proximidade e plausibilidade
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
