import numpy as np
from typing import List, Tuple, Dict, Optional
from .metrics import dtw_distance, plausibility_penalty

def evaluate_objectives_batch(r0: np.ndarray, pop: np.ndarray, model, cfg):
    X = pop[:, :, None]
    P = model.predict_proba(X)
    neg_conf = -P[:, cfg.target_class]
    prox = np.zeros(len(pop), dtype=float)
    plau = np.zeros(len(pop), dtype=float)
    spars = np.zeros(len(pop), dtype=float)
    for i in range(len(pop)):
        prox[i]  = dtw_distance(pop[i], r0, radius=getattr(cfg, 'fastdtw_radius', 1))
        plau[i]  = plausibility_penalty(pop[i], r0, cfg)
        spars[i] = np.abs(pop[i] - r0).sum()
    return np.vstack([neg_conf, prox, plau, spars]).T

def non_dominated_sort(F: List[List[float]]) -> List[List[int]]:
    n = len(F); S = [set() for _ in range(n)]; n_dom = [0]*n; fronts=[[]]
    for p in range(n):
        for q in range(n):
            if p==q: continue
            if all(F[p][i] <= F[q][i] for i in range(len(F[0]))) and any(F[p][i] < F[q][i] for i in range(len(F[0]))):
                S[p].add(q)
            elif all(F[q][i] <= F[p][i] for i in range(len(F[0]))) and any(F[q][i] < F[p][i] for i in range(len(F[0]))):
                n_dom[p]+=1
        if n_dom[p]==0: fronts[0].append(p)
    i=0
    while fronts[i]:
        nxt=[]
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q]-=1
                if n_dom[q]==0: nxt.append(q)
        i+=1; fronts.append(nxt)
    return fronts[:-1]

def crowding_distance(front_vals: List[List[float]], idxs: List[int]) -> Dict[int,float]:
    m = len(front_vals[0]); cd = {idxs[i]:0.0 for i in range(len(front_vals))}
    for k in range(m):
        vals = np.array([front_vals[i][k] for i in range(len(front_vals))])
        order = np.argsort(vals)
        cd[idxs[order[0]]]  = float('inf')
        cd[idxs[order[-1]]] = float('inf')
        minv, maxv = vals[order[0]], vals[order[-1]]
        denom = max(maxv - minv, 1e-12)
        for j in range(1, len(order)-1):
            prevv = vals[order[j-1]]; nextv = vals[order[j+1]]
            cd[idxs[order[j]]] += (nextv - prevv)/denom
    return cd

def nsga2_step(pop: np.ndarray, objs: List[List[float]], cfg) -> Tuple[np.ndarray, List[List[float]]]:
    fronts = non_dominated_sort(objs)
    elite_idxs = []
    for fr in fronts:
        if len(elite_idxs) + len(fr) <= cfg.population_size:
            elite_idxs.extend(fr)
        else:
            front_vals = [objs[i] for i in fr]
            cds = crowding_distance(front_vals, fr)
            order = sorted(fr, key=lambda i: -cds[i])
            elite_idxs.extend(order[:cfg.population_size - len(elite_idxs)])
            break
    pop = pop[elite_idxs]

    from .operators import crossover_one_point, mutate
    offspring = []
    while len(offspring) < cfg.population_size:
        i, j = np.random.randint(0, len(pop), size=2)
        a, b = pop[i].copy(), pop[j].copy()
        if np.random.rand() < cfg.crossover_rate:
            a, b = crossover_one_point(a, b)
        if np.random.rand() < cfg.mutation_rate: a = mutate(a, cfg)
        if np.random.rand() < cfg.mutation_rate: b = mutate(b, cfg)
        offspring.extend([a, b])
    pop_next = np.array(offspring[:cfg.population_size])
    return pop_next, None

def run_ebits(r0_batch: np.ndarray, model, cfg, rng=True, bank: Optional[np.ndarray]=None):
    if rng:
        np.random.seed(cfg.random_seed)
    from .datasets import initial_population_from_bank
    results = []
    for b in range(len(r0_batch)):
        r0 = r0_batch[b]
        if bank is not None:
            pop = initial_population_from_bank(r0, bank, pop_size=cfg.population_size, noise_sigma=0.01)
        else:
            pop = np.stack([r0 + 0.01*np.random.randn(len(r0)) for _ in range(cfg.population_size)])
        objs = evaluate_objectives_batch(r0, pop, model, cfg)
        history = []
        for gen in range(cfg.n_generations):
            pop, _ = nsga2_step(pop, objs, cfg)
            objs = evaluate_objectives_batch(r0, pop, model, cfg)
            best_idx = int(np.argmin(objs[:,0]))
            history.append((gen, pop[best_idx].copy(), objs[best_idx]))
        fronts = non_dominated_sort(objs.tolist())
        pareto_idx = fronts[0]
        results.append({"r0": r0, "pop": pop, "objs": np.array(objs),
                        "pareto_idx": np.array(pareto_idx), "history": history})
    return results