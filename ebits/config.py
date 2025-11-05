from dataclasses import dataclass

@dataclass
class EBITSConfig:
    window_len: int = 256
    horizon: int = 20
    population_size: int = 48
    n_generations: int = 40
    crossover_rate: float = 0.8
    mutation_rate: float = 0.9
    target_class: int = 1  # 1 = subida
    use_multiobjective: bool = True
    max_abs_jump: float = 5.0    # em sigmas de retorno
    max_global_scale: float = 2.0
    acf_max_lag: int = 20
    tournament_k: int = 2
    random_seed: int = 42
