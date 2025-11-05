from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

@dataclass
class PatchTSTConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_ff: int = 256
    dropout: float = 0.1
    patch_len: int = 16
    patch_stride: int = 8
    cls_token: bool = True
    # treino
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "cpu"  # "cuda" se disponível

class Patchify(nn.Module):
    """Converte sequência (B, L, C) em sequência de patches (B, P, patch_len*C)."""
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        # unfold ao longo de L
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # (B, C, P, patch_len)
        B, C, P, PL = patches.shape
        patches = patches.permute(0, 2, 1, 3).contiguous()  # (B, P, C, patch_len)
        patches = patches.view(B, P, C*PL)                  # (B, P, C*patch_len)
        return patches

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T]

class PatchTST(nn.Module):
    def __init__(self, input_channels: int, n_classes: int, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        self.patchify = Patchify(cfg.patch_len, cfg.patch_stride)
        # embed de patch linear
        self.proj = nn.Linear(input_channels * cfg.patch_len, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, batch_first=True, activation='gelu'
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model)) if cfg.cls_token else None
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, n_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        patches = self.patchify(x)           # (B, P, C*patch_len)
        z = self.proj(patches)               # (B, P, D)
        if self.cls_token is not None:
            cls = self.cls_token.expand(z.size(0), -1, -1)  # (B,1,D)
            z = torch.cat([cls, z], dim=1)   # (B, 1+P, D)
        z = self.pos(z)
        z = self.enc(z)                      # (B, 1+P, D)
        z = self.norm(z)
        if self.cls_token is not None:
            pooled = z[:, 0]                 # (B, D)
        else:
            pooled = z.mean(dim=1)           # (B, D)
        logits = self.head(pooled)           # (B, n_classes)
        return logits

class TorchUpDown:
    """Wrapper compatível com EBITS: expõe predict_proba(X) com X (B, L, 1)."""
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:  # (B, L) -> (B, L, 1)
            X = X[..., None]
        x = torch.from_numpy(X).float().to(self.device)
        logits = self.model(x)
        P = F.softmax(logits, dim=-1).cpu().numpy()
        return P

def train_patchtst_on_windows(Xtr: np.ndarray, ytr: np.ndarray,
                              Xva: np.ndarray, yva: np.ndarray,
                              model_cfg: PatchTSTConfig) -> TorchUpDown:
    device = torch.device(model_cfg.device)
    n_classes = int(np.max(ytr)) + 1
    model = PatchTST(input_channels=Xtr.shape[-1], n_classes=n_classes, cfg=model_cfg).to(device)

    # Datasets
    def to_tensor(x):
        if x.ndim == 2:
            x = x[..., None]
        return torch.from_numpy(x).float()
    tr_ds = TensorDataset(to_tensor(Xtr), torch.from_numpy(ytr).long())
    va_ds = TensorDataset(to_tensor(Xva), torch.from_numpy(yva).long())
    tr_loader = DataLoader(tr_ds, batch_size=model_cfg.batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=model_cfg.batch_size, shuffle=False, drop_last=False)

    # Otimizador e loss
    opt = torch.optim.AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)
    crit = nn.CrossEntropyLoss()

    best_va = -1.0
    best_state = None
    for epoch in range(model_cfg.epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        # validação
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                total   += int(yb.numel())
        va_acc = correct / max(1,total)
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return TorchUpDown(model, device=str(device))
