import argparse, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ebits.config import EBITSConfig, apply_profile
from ebits.datasets import load_ohlcv_csv, compute_log_returns, make_windows, zscore_per_asset, feature_block
from ebits.datasets import build_window_bank
from ebits.evolution import run_ebits
from ebits.metrics import compute_summary_metrics
from ebits.utils import cluster_patterns

def series_from_csv(csv_path):
    import pandas as pd
    df = load_ohlcv_csv(csv_path)
    r = compute_log_returns(df[[c for c in df.columns if c.lower()=='close'][0]])
    return zscore_per_asset(r)

def prep_random_series(T=6000, drift=0.0002, vol=0.01):
    prices = np.cumprod(1 + np.random.randn(T)*vol + drift) * 100.0
    r = np.diff(np.log(prices))
    return zscore_per_asset(r)

def make_dataset(r, L, H, step):
    X_raw, y, _ = make_windows(r, L=L, step=step, horizon=H)
    return X_raw, y

def train_model(Xtr, ytr, Xte, yte, model_kind, device="cpu", focal=False):
    if model_kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from ebits.models_iface import SKLearnUpDown
        Ftr = feature_block(Xtr); Fte = feature_block(Xte)
        clf = LogisticRegression(max_iter=1000).fit(Ftr, ytr)
        model = SKLearnUpDown(clf, transform_fn=feature_block, n_classes=2)
        acc = accuracy_score(yte, clf.predict(Fte))
        print(f"Acurácia baseline (LogReg): {acc:.3f}")
        return model, acc
    else:
        from ebits.models_patchtst import PatchTSTConfig, train_patchtst_on_windows
        Xtr_t, Xte_t = Xtr[..., None], Xte[..., None]
        cfg_t = PatchTSTConfig(device=device, focal_loss=focal)
        model = train_patchtst_on_windows(Xtr_t, ytr, Xte_t, yte, cfg_t)
        from ebits.models_patchtst import torch
        with torch.no_grad():
            P = model.predict_proba(Xte_t)
        acc = accuracy_score(yte, P.argmax(axis=1))
        print(f"Acurácia (PatchTST): {acc:.3f}")
        return model, acc

def run_profile(profile_name, args, bank_dir=None):
    if profile_name == "intraday":
        L, H, STEP = (args.intra_L or 256), (args.intra_H or 20), (args.intra_step or 5)
    else:
        L, H, STEP = (args.daily_L or 256), (args.daily_H or 5), (args.daily_step or 1)

    if args.csv:
        r = series_from_csv(args.csv)
    else:
        r = prep_random_series()
    X_raw, y = make_dataset(r, L, H, STEP)
    Xtr, Xte, ytr, yte = train_test_split(X_raw, y, test_size=0.3, shuffle=False)

    model, acc = train_model(Xtr, ytr, Xte, yte, args.model, device=args.device, focal=args.focal)

    bank = None
    if bank_dir:
        print(f"[{profile_name}] Construindo window-bank real de: {bank_dir}")
        bank = build_window_bank(bank_dir, L=L, step=STEP, horizon=H, limit_files=200)
        print(f"[{profile_name}] Window-bank pronto: {bank.shape} janelas")

    cfg = EBITSConfig(window_len=L, horizon=H,
                      n_generations=args.generations, population_size=args.pop_size,
                      use_fastdtw=True, fastdtw_radius=args.fastdtw_radius,
                      max_abs_jump=3.0,
                      w_acf=1.0, w_acf2=1.0, w_skewkurt=0.5, w_jb=0.25, w_spectrum=1.0)
    cfg = apply_profile(cfg, profile_name)

    B = min(args.n_ebits, len(Xte))
    r0_batch = Xte[:B]
    results = run_ebits(r0_batch, model, cfg, rng=True, bank=bank)
    metrics = compute_summary_metrics(model, cfg.target_class, r0_batch, results)
    print(f"=== Métricas EBITS ({profile_name}) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    all_pareto = []
    for res in results:
        Pset = res["pop"][res["pareto_idx"]]
        all_pareto.append(Pset)
    if all_pareto:
        import matplotlib.pyplot as plt
        P = np.vstack(all_pareto)
        centers, labels = cluster_patterns(P, n_clusters=5)
        print(f"[{profile_name}] Protótipos (5) gerados. centers shape:", centers.shape)
        os.makedirs("figs", exist_ok=True)
        plt.figure(figsize=(8,4))
        for i in range(centers.shape[0]):
            plt.plot(centers[i], label=f'proto {i+1}')
        plt.title(f"Protótipos (centroids) do Pareto set - {profile_name}")
        plt.legend(); plt.tight_layout()
        out = os.path.join("figs", f"prototipos_pareto_{profile_name}.png")
        plt.savefig(out)
        print(f"Figura salva: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="", help="CSV com 'close' (opcional; se vazio, série sintética)")
    ap.add_argument("--model", type=str, default="patchtst", choices=["logreg", "patchtst"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--n_ebits", type=int, default=12)
    ap.add_argument("--pop_size", type=int, default=48)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--fastdtw_radius", type=int, default=1)
    ap.add_argument("--bank_intraday", type=str, default="", help="Pasta com CSVs intraday")
    ap.add_argument("--bank_daily", type=str, default="", help="Pasta com CSVs diário/semanal")
    ap.add_argument("--intra_L", type=int, default=None)
    ap.add_argument("--intra_H", type=int, default=None)
    ap.add_argument("--intra_step", type=int, default=None)
    ap.add_argument("--daily_L", type=int, default=None)
    ap.add_argument("--daily_H", type=int, default=None)
    ap.add_argument("--daily_step", type=int, default=None)
    args = ap.parse_args()

    if args.bank_intraday:
        run_profile("intraday", args, bank_dir=args.bank_intraday)
    if args.bank_daily:
        run_profile("daily", args, bank_dir=args.bank_daily)
    if not args.bank_intraday and not args.bank_daily:
        run_profile("daily", args, bank_dir=None)

if __name__ == "__main__":
    main()