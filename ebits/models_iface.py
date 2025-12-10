from typing import Protocol, Callable
import numpy as np

class BlackBoxModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

class SKLearnUpDown:
    def __init__(self, model, transform_fn: Callable[[np.ndarray], np.ndarray], n_classes: int = 2):
        self.model = model
        self.transform_fn = transform_fn
        self.n_classes = n_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3 and X.shape[-1] == 1:
            Xw = X[..., 0]
        elif X.ndim == 2:
            Xw = X
        else:
            raise ValueError("Entrada deve ser (batch, L, 1) ou (batch, L)")
        feats = self.transform_fn(Xw)
        P = self.model.predict_proba(feats)
        return P