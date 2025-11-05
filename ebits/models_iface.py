from typing import Protocol, Callable, Optional
import numpy as np

class BlackBoxModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

class SKLearnUpDown:
    """Wrapper para modelos sklearn com transform de features.
    transform_fn: recebe X_raw (batch, L) e retorna matriz (batch, D).
    model: objeto sklearn com .predict_proba
    n_classes: normalmente 2 (down/up).
    """
    def __init__(self, model, transform_fn: Callable[[np.ndarray], np.ndarray], n_classes: int = 2):
        self.model = model
        self.transform_fn = transform_fn
        self.n_classes = n_classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # X esperado: (batch, L, 1) com retornos na última dimensão
        if X.ndim == 3 and X.shape[-1] == 1:
            Xw = X[..., 0]
        elif X.ndim == 2:
            Xw = X
        else:
            raise ValueError("Entrada deve ser (batch, L, 1) ou (batch, L)")
        feats = self.transform_fn(Xw)
        P = self.model.predict_proba(feats)
        # garantir shape (batch, n_classes)
        return P
