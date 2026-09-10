"""
knn_accuracy_model.py — Reusable KNN accuracy regressor
=========================================================

A simple, interpretable non-parametric regressor for the accuracy-compression
function A(eta). Uses k-nearest neighbours with inverse-distance weighting.

Advantages over the neural AccuracyPredictor for visualisation and analysis:
  - No training required — fit is O(N) (just stores the data)
  - Prediction is exact at observed points
  - Smoothness controlled by a single interpretable parameter k
  - Works well for concavity analysis where the landscape may be non-concave
    (the neural predictor's architecture forces concavity which is wrong below
    eta_min^true)

The model can be saved/loaded as a simple JSON or npz file and reused across
scripts (3D plots, concavity analysis, paper figures).

Usage
-----
    from src.core.knn_accuracy_model import KNNAccuracyModel

    # Fit
    model = KNNAccuracyModel(k=5, distance='euclidean')
    model.fit(eta_grid, acc_values)           # eta_grid: (N, D), acc_values: (N,)
    model.save('knn_resnet_topk.npz')

    # Predict on new points
    acc_pred = model.predict(eta_new)         # eta_new: (M, D) -> (M,)

    # Load saved model
    model2 = KNNAccuracyModel.load('knn_resnet_topk.npz')
"""

from __future__ import annotations

import json
import os
from typing import Optional, Literal

import numpy as np


class KNNAccuracyModel:
    """
    K-Nearest Neighbour accuracy regressor with inverse-distance weighting.

    Parameters
    ----------
    k : int
        Number of neighbours. Larger k = smoother surface.
        k=1 → interpolates exactly through data points (noisy).
        k=5 → good balance of smoothness and fidelity for typical grids.
        k=10-20 → very smooth, useful for paper-quality figures.
    distance : str
        Distance metric: 'euclidean' (default) or 'cosine'.
    weighting : str
        'inverse_distance' (default) — weight = 1/d (exact at data points).
        'uniform' — simple average of k nearest neighbours.
    epsilon : float
        Small constant added to distances to avoid division by zero.
    """

    def __init__(
        self,
        k:          int   = 5,
        distance:   Literal['euclidean', 'cosine'] = 'euclidean',
        weighting:  Literal['inverse_distance', 'uniform'] = 'inverse_distance',
        epsilon:    float = 1e-8,
    ):
        self.k         = k
        self.distance  = distance
        self.weighting = weighting
        self.epsilon   = epsilon
        self._X: Optional[np.ndarray] = None  # (N, D) training eta vectors
        self._y: Optional[np.ndarray] = None  # (N,)  training accuracy values
        self._fitted = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNAccuracyModel":
        """
        Store training data.

        Parameters
        ----------
        X : (N, D) array of eta vectors
        y : (N,)   array of accuracy values
        """
        self._X = np.array(X, dtype=np.float64)
        self._y = np.array(y, dtype=np.float64)
        self._fitted = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """
        Predict accuracy for query points.

        Parameters
        ----------
        X_query : (M, D) or (D,) array of eta vectors

        Returns
        -------
        y_pred : (M,) or scalar array of predicted accuracies
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        scalar = X_query.ndim == 1
        if scalar:
            X_query = X_query[np.newaxis, :]
        X_query = np.array(X_query, dtype=np.float64)

        M   = X_query.shape[0]
        k   = min(self.k, len(self._X))
        out = np.zeros(M)

        for i in range(M):
            dists = self._distances(X_query[i], self._X)
            idx   = np.argpartition(dists, k)[:k]
            d_k   = dists[idx]
            y_k   = self._y[idx]

            if self.weighting == 'inverse_distance':
                w   = 1.0 / (d_k + self.epsilon)
                out[i] = float(np.sum(w * y_k) / np.sum(w))
            else:
                out[i] = float(np.mean(y_k))

        return out[0] if scalar else out

    # ── Gradient (finite difference) ─────────────────────────────────────────

    def gradient(self, x: np.ndarray, delta: float = 0.02) -> np.ndarray:
        """
        Estimate gradient at point x via central finite differences.

        Parameters
        ----------
        x     : (D,) eta vector
        delta : finite difference step size

        Returns
        -------
        grad : (D,) gradient vector
        """
        x   = np.array(x, dtype=np.float64)
        D   = len(x)
        grad = np.zeros(D)
        for i in range(D):
            xp, xm = x.copy(), x.copy()
            xp[i] = min(1.0, x[i] + delta)
            xm[i] = max(0.0, x[i] - delta)
            step   = xp[i] - xm[i]
            if step > 0:
                grad[i] = float(
                    (self.predict(xp) - self.predict(xm)) / step
                )
        return grad

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model to .npz file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            X         = self._X,
            y         = self._y,
            k         = np.array([self.k]),
            distance  = np.array([self.distance]),
            weighting = np.array([self.weighting]),
            epsilon   = np.array([self.epsilon]),
        )
        print(f"[KNNAccuracyModel] Saved to {path}  "
              f"(N={len(self._X)}, k={self.k})", flush=True)

    @classmethod
    def load(cls, path: str) -> "KNNAccuracyModel":
        """Load model from .npz file."""
        data = np.load(path, allow_pickle=True)
        model = cls(
            k         = int(data['k'][0]),
            distance  = str(data['distance'][0]),
            weighting = str(data['weighting'][0]),
            epsilon   = float(data['epsilon'][0]),
        )
        model._X      = data['X']
        model._y      = data['y']
        model._fitted = True
        print(f"[KNNAccuracyModel] Loaded from {path}  "
              f"(N={len(model._X)}, k={model.k})", flush=True)
        return model

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _distances(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Compute distances from x to all rows of X."""
        if self.distance == 'euclidean':
            diff = X - x[np.newaxis, :]
            return np.sqrt(np.sum(diff**2, axis=1))
        elif self.distance == 'cosine':
            xn  = x / (np.linalg.norm(x) + self.epsilon)
            Xn  = X / (np.linalg.norm(X, axis=1, keepdims=True) + self.epsilon)
            return 1.0 - Xn @ xn
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def leave_one_out_rmse(self) -> float:
        """
        Leave-one-out cross-validation RMSE on training data.
        Useful for selecting k without a separate validation set.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        N  = len(self._X)
        sq = 0.0
        for i in range(N):
            # Temporarily exclude point i
            X_loo = np.delete(self._X, i, axis=0)
            y_loo = np.delete(self._y, i)
            m_loo = KNNAccuracyModel(
                k=self.k, distance=self.distance,
                weighting=self.weighting, epsilon=self.epsilon
            ).fit(X_loo, y_loo)
            pred  = float(m_loo.predict(self._X[i]))
            sq   += (pred - self._y[i]) ** 2
        return float(np.sqrt(sq / N))

    def summary(self) -> str:
        if not self._fitted:
            return "KNNAccuracyModel (not fitted)"
        return (f"KNNAccuracyModel(k={self.k}, N={len(self._X)}, "
                f"D={self._X.shape[1]}, "
                f"A_range=[{self._y.min():.4f}, {self._y.max():.4f}])")
