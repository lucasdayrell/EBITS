import argparse, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ebits.config import EBITSConfig
from ebits.datasets import load_ohlcv_csv, compute_log_returns, make_windows, zscore_per_asset, feature_block
from ebits.models_iface import SKLearnUpDown
from ebits.evolution import run_ebits
from ebits.metrics import compute_summary_metrics
from ebits.utils import cluster_patterns

def prepare_data(csv_path, L, horizon, step):
    if csv_path and os.path.exists(csv_path):
        import pandas as pd
        df = load_ohlcv_csv(csv_path)
        r = compute_log_returns(df[[c for c in df.columns if c.lower()=='close'][0]])
    else:
        # série sintética
        T = 6000
        drift = 0.0002; vol = 0.01
        prices = np.cumprod(1 + np.random.randn(T)*vol + drift) * 100.0
        r = np.diff(np.log(prices))
    r = zscore_per_asset(r)
    X_raw, y, idxs = make_windows(r, L=L, step=step, horizon=horizon)
    return X_raw, y, idxs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="", help="Caminho para CSV com OHLCV (usa close)")
    ap.add_argument("--window_len", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--n_ebits", type=int, default=12, help="Quantas janelas do teste usar no EBITS")
    ap.add_argument("--model", type=str, default="logreg", choices=["logreg", "patchtst"])
    ap.add_argument("--device", type=str, default="cpu", help="cpu ou cuda")
    args = ap.parse_args()

    L, H = args.window_len, args.horizon
    X_raw, y, _ = prepare_data(args.csv, L, H, args.step)
    Xtr, Xte, ytr, yte = train_test_split(X_raw, y, test_size=0.3, shuffle=False)

    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression
        Ftr = feature_block(Xtr); Fte = feature_block(Xte)
        clf = LogisticRegression(max_iter=1000).fit(Ftr, ytr)
        from ebits.models_iface import SKLearnUpDown
        model = SKLearnUpDown(clf, transform_fn=feature_block, n_classes=2)
        pred = clf.predict(Fte)
        acc = accuracy_score(yte, pred)
        print(f"Acurácia baseline (LogReg): {acc:.3f}")
    else:
        # PatchTST
        from ebits.models_patchtst import PatchTSTConfig, train_patchtst_on_windows
        # Ajuste rápido: adicionar canal (feature) 1
        Xtr_t = Xtr[..., None]; Xte_t = Xte[..., None]
        cfg_t = PatchTSTConfig(device=args.device, epochs=15, d_model=128, n_heads=4, n_layers=3, dim_ff=256,
                               patch_len=16, patch_stride=8, dropout=0.1, batch_size=128, lr=1e-3, weight_decay=1e-4)
        model = train_patchtst_on_windows(Xtr_t, ytr, Xte_t, yte, cfg_t)
        # medir acurácia do modelo treinado
        from ebits.models_patchtst import torch, F
        with torch.no_grad():
            P = model.predict_proba(Xte_t)
        acc = accuracy_score(yte, P.argmax(axis=1))
        print(f"Acurácia (PatchTST): {acc:.3f}")

    # EBITS
    B = min(args.n_ebits, len(Xte))
    r0_batch = Xte[:B]  # shape (B, L)
    cfg = EBITSConfig(window_len=L, horizon=H, n_generations=30, population_size=40)
    results = run_ebits(r0_batch, model, cfg, rng=True)
    metrics = compute_summary_metrics(model, cfg.target_class, r0_batch, results)
    print("=== Métricas EBITS v1 ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Protótipos
    all_pareto = []
    for res in results:
        Pset = res["pop"][res["pareto_idx"]]
        all_pareto.append(Pset)
    if all_pareto:
        import matplotlib.pyplot as plt
        P = np.vstack(all_pareto)
        centers, labels = cluster_patterns(P, n_clusters=5)
        print("Protótipos (5) gerados. centers shape:", centers.shape)
        plt.figure(figsize=(8,4))
        for i in range(centers.shape[0]):
            plt.plot(centers[i], label=f'proto {i+1}')
        plt.title("Protótipos (centroids) do Pareto set")
        plt.legend()
        plt.tight_layout()
        plt.savefig("prototipos_pareto.png")
        print("Figura salva: prototipos_pareto.png")

if __name__ == "__main__":
    main()
