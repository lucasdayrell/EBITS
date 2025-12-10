from dataclasses import dataclass, field
from typing import Dict

@dataclass
class EBITSConfig:
    window_len: int = 256
    horizon: int = 20
    population_size: int = 48
    n_generations: int = 40
    crossover_rate: float = 0.8
    mutation_rate: float = 0.9
    target_class: int = 1
    use_multiobjective: bool = True
    tournament_k: int = 2
    random_seed: int = 42

    # DTW
    use_fastdtw: bool = True
    fastdtw_radius: int = 1

    # limites/plausibilidade
    max_abs_jump: float = 3.0
    max_global_scale: float = 2.0
    acf_max_lag: int = 20

    # pesos plausibilidade
    w_acf: float = 1.0
    w_acf2: float = 1.0
    w_skewkurt: float = 0.5
    w_jb: float = 0.25
    w_spectrum: float = 0.5

    # mix de operadores (probabilidades) e suas forças
    op_probs: Dict[str, float] = field(default_factory=lambda: {
        "impulse": 0.25, "trend": 0.25, "vol": 0.25, "warp": 0.25
    })
    op_strength: Dict[str, float] = field(default_factory=lambda: {
        "impulse": 1.0, "trend": 0.5, "vol": 1.0, "warp": 0.2
    })

def apply_profile(cfg: 'EBITSConfig', profile: str) -> 'EBITSConfig':
    p = profile.lower()
    if p in ("intraday", "intra"):
        cfg.acf_max_lag = 30
        cfg.w_acf = 0.6; cfg.w_acf2 = 1.2
        cfg.w_skewkurt = 0.6; cfg.w_jb = 0.2
        cfg.w_spectrum = 1.2
        cfg.max_abs_jump = 4.0
        cfg.op_strength.update({"impulse": 0.8, "trend": 0.4, "vol": 1.2, "warp": 0.25})
    elif p in ("daily", "diario", "weekly", "semanal"):
        cfg.acf_max_lag = 20
        cfg.w_acf = 1.0; cfg.w_acf2 = 0.8
        cfg.w_skewkurt = 0.8; cfg.w_jb = 0.4
        cfg.w_spectrum = 1.0
        cfg.max_abs_jump = 3.0
        cfg.op_strength.update({"impulse": 0.7, "trend": 0.6, "vol": 0.9, "warp": 0.2})
    return cfg