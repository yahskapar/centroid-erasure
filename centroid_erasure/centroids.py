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

import hashlib
import json
import os
import tempfile
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
    def load(cls, path, modality="text", expected_model=None):
        """Load a bank from a .npz written by `save_pair` or the paper pipeline.

        Args:
            path: path to the .npz file.
            modality: "text" or "visual" — selects which array to read.
            expected_model: optional registry key. When supplied, fail closed
                unless the bank embeds matching provenance or is a checksummed
                shipped bank bound to that model in ``centroids/MANIFEST.json``.
        """
        key = {"text": "text_centroids", "visual": "vis_centroids"}[modality]
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(
                    f"{path} has no array '{key}'. Present: {list(data.files)}"
                )
            centers = data[key]
            persisted = {}
            if "metadata_json" in data.files:
                try:
                    pair_meta = json.loads(str(data["metadata_json"].item()))
                    persisted = pair_meta.get(modality, {})
                except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as e:
                    raise ValueError(f"{path} has invalid metadata_json: {e}") from e
                if not isinstance(persisted, dict):
                    raise ValueError(
                        f"{path} metadata for {modality!r} must be an object"
                    )
        manifest_meta = _manifest_provenance(path, modality)
        for field, value in manifest_meta.items():
            if field in persisted and persisted[field] != value:
                raise ValueError(
                    f"{path} metadata field {field!r} conflicts with its "
                    "checksummed manifest record"
                )
            persisted.setdefault(field, value)
        recorded_modality = persisted.get("modality")
        if recorded_modality is not None and recorded_modality != modality:
            raise ValueError(
                f"{path} metadata says modality={recorded_modality!r}, "
                f"but {modality!r} was requested"
            )
        meta = dict(persisted)
        meta["path"] = str(path)
        meta.setdefault("modality", modality)
        if expected_model is not None:
            required = {"model", "model_id", "model_revision", "layer"}
            missing = sorted(required - set(meta))
            if missing:
                raise ValueError(
                    f"{path} has incomplete model provenance (missing {missing}). "
                    "Use a bank fitted by `main.py fit`, or a checksummed bank "
                    "listed in centroids/MANIFEST.json."
                )
            recorded_model = meta["model"]
            if recorded_model != expected_model:
                raise ValueError(
                    f"{path} is bound to model {recorded_model!r}, not "
                    f"{expected_model!r}"
                )
        return cls(centers, meta=meta)

    @staticmethod
    def save_pair(path, text_bank, visual_bank):
        """Write a text/visual bank pair plus JSON provenance metadata.

        Existing paper artifacts contain only the two centroid arrays and remain
        fully supported. Newly fitted banks embed each modality's fit metadata
        as a non-pickled JSON scalar so the K-means backend and harvest
        provenance survive a save/load round trip.

        The write is atomic within the destination directory: an interrupted
        fit cannot leave a partially written bank at the requested path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "format_version": 1,
                "text": text_bank.meta,
                "visual": visual_bank.meta,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".npz", dir=path.parent, delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                np.savez(
                    tmp,
                    text_centroids=text_bank.mu.detach().cpu().numpy().astype(np.float32),
                    vis_centroids=visual_bank.mu.detach().cpu().numpy().astype(np.float32),
                    metadata_json=np.asarray(metadata),
                )
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()


def _json_default(value):
    """Convert common provenance scalar types without enabling pickle."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"metadata value {value!r} is not JSON serializable")


def _manifest_provenance(path, modality):
    """Return verified provenance for a legacy shipped bank, if applicable.

    The original paper banks predate embedded ``metadata_json``. Their
    provenance therefore lives in the adjacent manifest. Merely matching the
    hidden width is not sufficient: several released models share a width.
    """
    manifest_path = path.parent / "MANIFEST.json"
    if not manifest_path.is_file():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read centroid manifest {manifest_path}: {exc}") from exc

    rel = f"{path.parent.name}/{path.name}"
    record = manifest.get("files", {}).get(rel)
    provenance = manifest.get("bank_provenance", {}).get(rel)
    if record is None or provenance is None:
        return {}

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record.get("sha256"):
        raise ValueError(
            f"{path} does not match its recorded SHA-256 in {manifest_path}"
        )

    layer_key = "text_layer" if modality == "text" else "visual_layer"
    return {
        "model": provenance["model"],
        "model_id": provenance["model_id"],
        "model_revision": provenance["model_revision"],
        "layer": provenance[layer_key],
        "manifest_sha256": digest,
    }


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
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError(f"X must have shape (N, D) with D > 0; got {X.shape}")
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    finite = np.isfinite(X).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        if verbose:
            print(f"    filtered {n_bad}/{len(X)} non-finite rows")
        X = X[finite]
    if len(X) < k:
        raise ValueError(
            f"cannot fit K={k} centroids from {len(X)} finite activation rows"
        )

    try:
        import faiss
    except ImportError:
        faiss = None

    # The CPU-only faiss wheel accepts ``gpu=True`` but silently resets the flag
    # and trains on CPU. Conversely, a real GPU training failure must propagate
    # instead of silently changing algorithms.
    if faiss is not None and faiss.get_num_gpus() > 0:
        km = faiss.Kmeans(X.shape[1], k, niter=20, gpu=True, seed=seed)
        km.train(X)
        if verbose:
            print(f"    fitted K={k} with faiss-gpu ({len(X)} tokens)")
        return CentroidBank(
            km.centroids,
            meta={
                "backend": "faiss-gpu",
                "backend_version": getattr(faiss, "__version__", "unknown"),
                "visible_faiss_gpus": faiss.get_num_gpus(),
                "k": k,
                "seed": seed,
                "n_tokens": len(X),
            },
        )

    import sklearn
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=k, random_state=seed, batch_size=4096, n_init=3
    )
    km.fit(X)
    if verbose:
        print(f"    fitted K={k} with sklearn ({len(X)} tokens)")
    return CentroidBank(
        km.cluster_centers_,
        meta={
            "backend": "sklearn",
            "backend_version": sklearn.__version__,
            "k": k,
            "seed": seed,
            "n_tokens": len(X),
        },
    )
