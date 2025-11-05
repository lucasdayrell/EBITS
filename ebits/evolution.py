import numpy as np
from typing import List, Tuple, Dict
from .metrics import dtw_distance, l2_distance, plausibility_penalty

def evaluate_objectives(r0: np.ndarray, r: np.ndarray, model, cfg) -> Tuple[float,float,float,float]:
    X = r[None, :, None]
    p = float(model.predict_proba(X)[0, cfg.target_class])
    neg_conf = -p
    prox = dtw_distance(r, r0)
    plau = plausibility_penalty(r, r0, cfg.acf_max_lag)
    spars = float(np.abs(r - r0).sum())
    return neg_conf, prox, plau, spars

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
    import numpy as np
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

    # reprodução
    offspring = []
    from .operators import crossover_one_point, mutate
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

def run_ebits(r0_batch: np.ndarray, model, cfg, rng=True):
    if rng:
        np.random.seed(cfg.random_seed)
    results = []
    for b in range(len(r0_batch)):
        r0 = r0_batch[b]
        pop = np.stack([r0 + 0.01*np.random.randn(len(r0)) for _ in range(cfg.population_size)])
        objs = [evaluate_objectives(r0, ind, model, cfg) for ind in pop]
        history = []
        for gen in range(cfg.n_generations):
            pop, _ = nsga2_step(pop, objs, cfg)
            objs = [evaluate_objectives(r0, ind, model, cfg) for ind in pop]
            best_idx = int(np.argmin([o[0] for o in objs]))
            history.append((gen, pop[best_idx].copy(), objs[best_idx]))
        fronts = non_dominated_sort(objs)
        pareto_idx = fronts[0]
        results.append({
            "r0": r0,
            "pop": pop,
            "objs": np.array(objs),
            "pareto_idx": np.array(pareto_idx),
            "history": history
        })
    return results
