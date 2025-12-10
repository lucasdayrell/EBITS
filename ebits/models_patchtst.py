from dataclasses import dataclass
import math, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

@dataclass
class PatchTSTConfig:
    d_model:int=128; n_heads:int=4; n_layers:int=3; dim_ff:int=256; dropout:float=0.1
    patch_len:int=16; patch_stride:int=8; cls_token:bool=True
    batch_size:int=128; lr:float=1e-3; weight_decay:float=1e-4; epochs:int=15
    device:str="cpu"
    focal_loss: bool=False; focal_gamma: float=2.0

class Patchify(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__(); self.patch_len=patch_len; self.stride=stride
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        B, C, P, PL = patches.shape
        patches = patches.permute(0,2,1,3).contiguous().view(B, P, C*PL)
        return patches

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class PatchTST(nn.Module):
    def __init__(self, input_channels: int, n_classes: int, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        self.patchify = Patchify(cfg.patch_len, cfg.patch_stride)
        self.proj = nn.Linear(input_channels * cfg.patch_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads,
                                           dim_feedforward=cfg.dim_ff, dropout=cfg.dropout,
                                           batch_first=True, activation='gelu')
        self.pos = PositionalEncoding(cfg.d_model)
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model)) if cfg.cls_token else None
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, n_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.patchify(x))
        if self.cls_token is not None:
            cls = self.cls_token.expand(z.size(0), -1, -1)
            z = torch.cat([cls, z], dim=1)
        z = self.pos(z); z = self.enc(z); z = self.norm(z)
        pooled = z[:,0] if self.cls_token is not None else z.mean(dim=1)
        return self.head(pooled)

class TorchUpDown:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device); self.device = device; self.model.eval()
    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2: X = X[..., None]
        x = torch.from_numpy(X).float().to(self.device)
        logits = self.model(x)
        return F.softmax(logits, dim=-1).cpu().numpy()

def _class_weights(y, n_classes):
    import numpy as _np
    counts = _np.bincount(y, minlength=n_classes).astype(float)
    counts[counts==0] = 1.0
    inv = 1.0/counts
    w = inv*(n_classes/inv.sum())
    return torch.tensor(w, dtype=torch.float32)

def _focal_loss(logits, targets, weights=None, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=weights, reduction='none')
    pt = torch.exp(-ce); return ((1-pt)**gamma * ce).mean()

def train_patchtst_on_windows(Xtr, ytr, Xva, yva, cfg: PatchTSTConfig) -> TorchUpDown:
    import numpy as _np
    device = torch.device(cfg.device)
    if Xtr.ndim==2: Xtr = Xtr[...,None]
    if Xva.ndim==2: Xva = Xva[...,None]
    n_classes = int(_np.max(ytr))+1
    model = PatchTST(input_channels=Xtr.shape[-1], n_classes=n_classes, cfg=cfg).to(device)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long()),
                    batch_size=cfg.batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(yva).long()),
                    batch_size=cfg.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weights = _class_weights(ytr, n_classes).to(device)
    best_va, best_state = -1.0, None
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = _focal_loss(logits, yb, weights, cfg.focal_gamma) if cfg.focal_loss \
                   else F.cross_entropy(logits, yb, weight=weights)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); correct=0; total=0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(-1)
                correct += int((pred==yb).sum()); total += int(yb.numel())
        va_acc = correct/max(1,total)
        if va_acc>best_va: best_va, best_state = va_acc, {k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return TorchUpDown(model, device=str(device))