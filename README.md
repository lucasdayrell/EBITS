# EBITS v1 (demo) + PatchTST
Pipeline mínimo e **rodável** para evoluir janelas de retornos e auditar um classificador up/down.
Agora com opção **PatchTST** (PyTorch).

## Instalação
```bash
python -m venv .venv && source .venv/bin/activate  # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
# para usar PatchTST, instale também o PyTorch adequado ao seu sistema:
# CPU (geral):    pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 12 (ex.):  pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Rodar com LogReg (rápido)
```bash
python run_experiment.py --model logreg
```

## Rodar com PatchTST (PyTorch)
```bash
python run_experiment.py --model patchtst --device cpu
# se tiver GPU:
python run_experiment.py --model patchtst --device cuda
```

## CSV próprio
O CSV deve ter coluna `close` (case-insensitive):
```bash
python run_experiment.py --csv path/para/seu.csv --window_len 256 --horizon 20 --step 5 --n_ebits 12 --model patchtst
```

## Saídas
- Acurácia do modelo escolhido (LogReg ou PatchTST).
- EBITS em `n_ebits` janelas do teste.
- Métricas: flip_rate, mean_delta_conf, mean_dtw, mean_l2, mean_plaus.
- `prototipos_pareto.png` com 5 protótipos (centroides) do Pareto.

## Notas rápidas
- **PatchTSTConfig** (em `ebits/models_patchtst.py`) controla `patch_len`, `patch_stride`, `d_model`, etc.
- O wrapper `TorchUpDown` expõe `.predict_proba` no shape `(batch, L, 1)` para encaixar no EBITS.
