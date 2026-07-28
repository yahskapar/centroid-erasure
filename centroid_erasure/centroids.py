"""
Centroid banks: fitting, storage, and the replacement operation itself.

A bank is a set of K cluster centers fitted to activations harvested at one
layer of one model. Replacement snaps each token to its nearest center, then
interpolates back toward the original by alpha_interp:

    replace(x) = mu_k + alpha_interp * (x - mu_k)

so alpha_interp=0 is full collapse onto the centroid (maximum erasure) and
alpha_interp=1 is the identity. This is the sign convention used throughout
the paper; note that it is the opposite of "erasure strength".
"""

from pathlib import Path

import numpy as np
import torch

DEFAULT_K = 256
DEFAULT_KMEANS_SEED = 42


class CentroidBank:
    """K cluster centers for one (model, layer, modality) triple.

    Args:
        centers: (K, D) array or tensor of cluster centers.
        meta: optional dict recording how the bank was fitted.
    """

    def __init__(self, centers, meta=None):
        if isinstance(centers, np.ndarray):
            centers = torch.tensor(centers, dtype=torch.float32)
        self.mu = centers
        self.meta = dict(meta or {})
        self._device = None

    # ── properties ──

    @property
    def k(self):
        return self.mu.shape[0]

    @property
    def dim(self):
        return self.mu.shape[1]

    def __repr__(self):
        return f"CentroidBank(K={self.k}, D={self.dim})"

    # ── device handling ──

    def to_device(self, device):
        if self._device != device:
            self.mu = self.mu.to(device)
            self._device = device
        return self

    # ── the intervention ──

    def replace(self, x, alpha_interp=0.0):
        """Snap each row of x to its nearest center, then interpolate back.

        Args:
            x: (N, D) float tensor of activations.
            alpha_interp: 0.0 = full collapse onto centroids (max erasure),
                1.0 = identity. The paper's deployment protocol uses 0.4.

        Returns:
            (N, D) tensor, same device and dtype as the computation input.
        """
        self.to_device(x.device)
        dists = torch.cdist(x.unsqueeze(0), self.mu.unsqueeze(0))[0]
        k_idx = dists.argmin(dim=1)
        mu_k = self.mu[k_idx]
        return mu_k + alpha_interp * (x - mu_k)

    # ── persistence ──

    @classmethod
    def load(cls, path, modality="text"):
        """Load a bank from a .npz written by `save_pair` or the paper pipeline.

        Args:
            path: path to the .npz file.
            modality: "text" or "visual" — selects which array to read.
        """
        key = {"text": "text_centroids", "visual": "vis_centroids"}[modality]
        with np.load(path) as data:
            if key not in data.files:
                raise KeyError(
                    f"{path} has no array '{key}'. Present: {list(data.files)}"
                )
            centers = data[key]
        return cls(centers, meta={"path": str(path), "modality": modality})

    @staticmethod
    def save_pair(path, text_bank, visual_bank):
        """Write a text/visual bank pair in the layout the paper artifacts use."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            text_centroids=text_bank.mu.cpu().numpy().astype(np.float32),
            vis_centroids=visual_bank.mu.cpu().numpy().astype(np.float32),
        )


def fit_centroids(X, k=DEFAULT_K, seed=DEFAULT_KMEANS_SEED, verbose=True):
    """Fit a CentroidBank to harvested activations.

    Uses faiss-gpu when available and falls back to sklearn MiniBatchKMeans.
    The two backends do not produce bit-identical centers, so the backend
    actually used is recorded in `bank.meta["backend"]`.

    Args:
        X: (N, D) array of activations.
        k: number of clusters. The paper uses 256.
        seed: K-means seed. The paper's primary fit uses 42.

    Returns:
        CentroidBank.
    """
    X = np.asarray(X, dtype=np.float32)
    finite = np.isfinite(X).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        if verbose:
            print(f"    filtered {n_bad}/{len(X)} non-finite rows")
        X = X[finite]

    try:
        import faiss

        km = faiss.Kmeans(X.shape[1], k, niter=20, gpu=True, seed=seed)
        km.train(X)
        if verbose:
            print(f"    fitted K={k} with faiss-gpu ({len(X)} tokens)")
        return CentroidBank(
            km.centroids,
            meta={"backend": "faiss", "k": k, "seed": seed, "n_tokens": len(X)},
        )
    except Exception:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=k, random_state=seed, batch_size=4096, n_init=3
        )
        km.fit(X)
        if verbose:
            print(f"    fitted K={k} with sklearn ({len(X)} tokens)")
        return CentroidBank(
            km.cluster_centers_,
            meta={"backend": "sklearn", "k": k, "seed": seed, "n_tokens": len(X)},
        )
