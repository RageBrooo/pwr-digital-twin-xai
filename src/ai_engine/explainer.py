"""
explainer.py -- SHAP feature attributions for the Random Forest baselines.

  ex = RFExplainer(model, feature_names)
  ex.explain(X_rows)                -> per row: predicted class, probability, top contributing features
  ex.global_importance(X)           -> mean |SHAP| per feature and class (DataFrame)
  save_importance_plot(df, path)    -> stacked horizontal bar chart with real class names

Handles both SHAP output formats (list of per-class arrays, or one (n, features, classes) array).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import shap


def class_values(shap_values, class_index: int) -> np.ndarray:
    """-> (n_rows, n_features) SHAP values for one class, whichever format shap returned"""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])
    arr = np.asarray(shap_values)
    return arr[:, :, class_index] if arr.ndim == 3 else arr


class RFExplainer:
    def __init__(self, model, feature_names: Sequence[str]):
        self.model, self.names = model, list(feature_names)
        self.classes = list(model.classes_)
        self.tree = shap.TreeExplainer(model)

    def _values(self, X: pd.DataFrame):
        return self.tree.shap_values(X[self.names], check_additivity=False)

    def explain(self, X: pd.DataFrame, top_k: int = 8) -> List[Dict]:
        sv = self._values(X)
        proba = self.model.predict_proba(X[self.names])
        out = []
        for i in range(len(X)):
            k = int(np.argmax(proba[i]))
            v = class_values(sv, k)[i]
            order = np.argsort(-np.abs(v))[:top_k]
            out.append(dict(predicted=self.classes[k], probability=float(proba[i, k]),
                            top=[(self.names[j], float(v[j]), float(X[self.names].iloc[i, j])) for j in order]))
        return out

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        sv = self._values(X)
        cols = {c: np.abs(class_values(sv, k)).mean(axis=0) for k, c in enumerate(self.classes)}
        df = pd.DataFrame(cols, index=self.names)
        return df.loc[df.sum(axis=1).sort_values(ascending=False).index]


def save_importance_plot(imp: pd.DataFrame, path: Path, top: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = imp.head(top).iloc[::-1]
    ax = d.plot(kind="barh", stacked=True, figsize=(9, 6.5), width=0.8)
    ax.set_xlabel("mean |SHAP value| (average impact on the class scores)")
    ax.set_title("Random Forest feature attributions (held-out runs)")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
