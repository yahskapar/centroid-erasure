"""Centroid bank math, the alpha_interp sign convention, and shipped artifacts."""

import glob
import os

import numpy as np
import pytest
import torch

from centroid_erasure import PAPER_PROTOCOL, CentroidBank, fit_centroids

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANKS = sorted(glob.glob(os.path.join(REPO, "centroids", "*.npz")))


@pytest.fixture
def bank():
    rng = np.random.default_rng(0)
    return CentroidBank(rng.standard_normal((32, 16)).astype(np.float32))


# ── the sign convention ──
# This is the single easiest thing in the codebase to invert by accident, and
# inverting it silently reverses the direction of every reported effect.


def test_alpha_one_is_exactly_identity(bank):
    x = torch.randn(50, 16)
    assert torch.allclose(bank.replace(x, alpha_interp=1.0), x, atol=1e-6)


def test_alpha_zero_snaps_every_row_onto_a_centroid(bank):
    x = torch.randn(50, 16)
    y = bank.replace(x, alpha_interp=0.0)
    # Every output row must BE one of the centroids, not merely near one.
    for row in y:
        assert (bank.mu == row).all(dim=1).any(), "alpha_interp=0 must land on a centroid"


def test_erasure_is_monotone_in_alpha(bank):
    """Distance from the original must shrink as alpha_interp rises toward 1."""
    x = torch.randn(50, 16)
    dists = [
        (bank.replace(x, alpha_interp=a) - x).norm().item()
        for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    assert dists == sorted(dists, reverse=True), f"not monotone: {dists}"
    assert dists[-1] == pytest.approx(0.0, abs=1e-5)


def test_replacement_is_nearest_centroid_not_arbitrary(bank):
    x = torch.randn(20, 16)
    y = bank.replace(x, alpha_interp=0.0)
    expected = bank.mu[torch.cdist(x, bank.mu).argmin(dim=1)]
    assert torch.equal(y, expected)


def test_a_row_that_is_already_a_centroid_is_unchanged(bank):
    x = bank.mu[[3, 7, 11]].clone()
    assert torch.allclose(bank.replace(x, alpha_interp=0.0), x, atol=1e-6)


# ── persistence ──


def test_save_load_round_trip(tmp_path, bank):
    bank.meta.update({"backend": "sklearn", "seed": 42})
    other = CentroidBank(
        torch.randn(32, 16), meta={"backend": "faiss-gpu", "seed": 7}
    )
    p = tmp_path / "rt.npz"
    CentroidBank.save_pair(p, bank, other)
    text = CentroidBank.load(p, "text")
    visual = CentroidBank.load(p, "visual")
    assert torch.allclose(text.mu, bank.mu, atol=1e-6)
    assert torch.allclose(visual.mu, other.mu, atol=1e-6)
    assert text.meta["backend"] == "sklearn" and text.meta["seed"] == 42
    assert visual.meta["backend"] == "faiss-gpu" and visual.meta["seed"] == 7
    with np.load(p, allow_pickle=False) as data:
        assert data["metadata_json"].ndim == 0


def test_load_rejects_recorded_modality_mismatch(tmp_path, bank):
    bank.meta["modality"] = "visual"
    p = tmp_path / "wrong_modality.npz"
    CentroidBank.save_pair(p, bank, CentroidBank(torch.randn(32, 16)))
    with pytest.raises(ValueError, match="metadata says modality"):
        CentroidBank.load(p, "text")


def test_load_rejects_an_unknown_modality(tmp_path, bank):
    p = tmp_path / "rt.npz"
    CentroidBank.save_pair(p, bank, bank)
    with pytest.raises(KeyError):
        CentroidBank.load(p, "audio")


# ── the shipped artifacts ──


def test_seven_banks_are_shipped():
    assert len(BANKS) == 7, f"expected 7 shipped banks, found {[os.path.basename(b) for b in BANKS]}"


def test_no_medgemma_bank_ships():
    """Deliberately withheld: Health AI Developer Foundations licensing."""
    assert not any("medgemma" in os.path.basename(b).lower() for b in BANKS)


@pytest.mark.parametrize("path", BANKS, ids=lambda p: os.path.basename(p))
def test_shipped_bank_is_well_formed(path):
    for modality in ("text", "visual"):
        b = CentroidBank.load(path, modality)
        assert b.k == PAPER_PROTOCOL["k"], f"{path} {modality}: K={b.k}, expected 256"
        assert b.mu.dtype == torch.float32
        assert torch.isfinite(b.mu).all(), f"{path} {modality} contains NaN or inf"
        # Degenerate fits collapse centroids onto each other.
        assert len(torch.unique(b.mu, dim=0)) > b.k * 0.9, "centroids largely duplicated"


@pytest.mark.parametrize("path", BANKS, ids=lambda p: os.path.basename(p))
def test_text_and_visual_banks_have_matching_width_but_differ(path):
    t = CentroidBank.load(path, "text")
    v = CentroidBank.load(path, "visual")
    assert t.dim == v.dim
    assert not torch.equal(t.mu, v.mu), "text and visual banks are identical, likely a copy bug"


# ── fitting ──


def test_fit_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 8)).astype(np.float32)
    a = fit_centroids(X, k=8, seed=42, verbose=False)
    b = fit_centroids(X, k=8, seed=42, verbose=False)
    assert torch.allclose(a.mu, b.mu, atol=1e-5)


def test_fit_records_which_backend_was_used():
    """faiss and sklearn give different centers; runs are only comparable within one."""
    X = np.random.default_rng(2).standard_normal((200, 8)).astype(np.float32)
    meta = fit_centroids(X, k=4, verbose=False).meta
    assert meta["backend"] in (
        "faiss-gpu",
        "sklearn",
    )
    assert meta["backend_version"]


def test_cpu_only_faiss_does_not_masquerade_as_the_paper_gpu_backend():
    try:
        import faiss
    except ImportError:
        pytest.skip("faiss is not installed")
    if faiss.get_num_gpus() > 0:
        pytest.skip("a real FAISS GPU backend is visible")

    X = np.random.default_rng(4).standard_normal((200, 8)).astype(np.float32)
    assert fit_centroids(X, k=4, verbose=False).meta["backend"] == "sklearn"


def test_visible_faiss_gpu_training_failure_is_not_silently_changed_to_sklearn(
    monkeypatch,
):
    import sys
    import types

    class BrokenKMeans:
        def __init__(self, *args, **kwargs):
            pass

        def train(self, X):
            raise RuntimeError("FAISS GPU training failed")

    fake_faiss = types.SimpleNamespace(
        __version__="test",
        get_num_gpus=lambda: 1,
        Kmeans=BrokenKMeans,
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    X = np.random.default_rng(6).standard_normal((200, 8)).astype(np.float32)
    with pytest.raises(RuntimeError, match="FAISS GPU training failed"):
        fit_centroids(X, k=4, verbose=False)


def test_fit_drops_non_finite_rows_rather_than_propagating_nan():
    X = np.random.default_rng(3).standard_normal((200, 8)).astype(np.float32)
    X[5] = np.nan
    X[9] = np.inf
    assert torch.isfinite(fit_centroids(X, k=4, verbose=False).mu).all()


@pytest.mark.parametrize(
    "X,k,match",
    [
        (np.empty((0, 8), dtype=np.float32), 4, "from 0 finite"),
        (np.ones((3, 8), dtype=np.float32), 4, "from 3 finite"),
        (np.ones((8,), dtype=np.float32), 4, "shape"),
        (np.ones((8, 2), dtype=np.float32), 0, "positive"),
    ],
)
def test_fit_rejects_invalid_or_insufficient_activation_matrices(X, k, match):
    with pytest.raises(ValueError, match=match):
        fit_centroids(X, k=k, verbose=False)


# ── artifact integrity manifest ──


def test_manifest_exists_and_covers_every_shipped_artifact():
    import json

    manifest = json.load(open(os.path.join(REPO, "centroids", "MANIFEST.json")))
    listed = set(manifest["files"])
    shipped = {os.path.relpath(b, REPO) for b in BANKS}
    shipped.update(
        os.path.relpath(path, REPO)
        for path in glob.glob(os.path.join(REPO, "demo", "fixtures", "*.json"))
    )
    assert shipped == listed, (
        f"manifest mismatch: unlisted={sorted(shipped - listed)}, "
        f"missing={sorted(listed - shipped)}"
    )


@pytest.mark.parametrize("path", BANKS, ids=lambda p: os.path.basename(p))
def test_shipped_bank_matches_its_recorded_checksum(path):
    """Detects corruption, an accidental refit, or an LFS smudge."""
    import hashlib
    import json

    manifest = json.load(open(os.path.join(REPO, "centroids", "MANIFEST.json")))
    rel = os.path.relpath(path, REPO)
    recorded = manifest["files"][rel]
    actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert actual == recorded["sha256"], f"{rel} does not match the manifest"
    assert os.path.getsize(path) == recorded["bytes"]


def test_manifest_protocol_matches_the_code():
    import json

    manifest = json.load(open(os.path.join(REPO, "centroids", "MANIFEST.json")))
    for key in ("k", "text_layer", "visual_layer", "data_seed", "kmeans_seed"):
        assert manifest["protocol"][key] == PAPER_PROTOCOL[key], f"manifest {key} drifted"


def test_manifest_pins_the_shipped_models_and_core_datasets():
    import json

    from centroid_erasure import MODEL_REGISTRY
    from centroid_erasure.data.blink import BLINK_REVISION
    from centroid_erasure.data.coco import COCO_REVISION

    manifest = json.load(open(os.path.join(REPO, "centroids", "MANIFEST.json")))
    shipped = {os.path.basename(path)[:-4] for path in BANKS}
    assert set(manifest["model_revisions"]) == shipped
    for name in shipped:
        revision = MODEL_REGISTRY[name].revision
        assert revision == manifest["model_revisions"][name]
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    assert manifest["protocol"]["blink_revision"] == BLINK_REVISION
    assert manifest["protocol"]["coco_revision"] == COCO_REVISION


def test_manifest_pins_every_auxiliary_dataset_loader():
    import json

    from centroid_erasure.data.cvbench import CVBENCH_REPO, CVBENCH_REVISION
    from centroid_erasure.data.medblink import MEDBLINK_REPO, MEDBLINK_REVISION
    from centroid_erasure.data.mmvp import MMVP_REPO, MMVP_REVISION
    from centroid_erasure.data.pope import POPE_REPO, POPE_REVISION
    from centroid_erasure.data.scienceqa import SCIENCEQA_REPO, SCIENCEQA_REVISION

    manifest = json.load(open(os.path.join(REPO, "centroids", "MANIFEST.json")))
    expected = {
        MEDBLINK_REPO: MEDBLINK_REVISION,
        CVBENCH_REPO: CVBENCH_REVISION,
        MMVP_REPO: MMVP_REVISION,
        POPE_REPO: POPE_REVISION,
        SCIENCEQA_REPO: SCIENCEQA_REVISION,
    }
    assert manifest["auxiliary_dataset_revisions"] == expected
    for revision in expected.values():
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)


def test_environment_files_match_the_recorded_paper_environment():
    import json

    manifest = json.load(open(os.path.join(REPO, "centroids", "MANIFEST.json")))
    recorded = manifest["environment"]
    requirements = open(os.path.join(REPO, "requirements.txt")).read()
    environment_yml = open(os.path.join(REPO, "environment.yml")).read()

    assert f"python={recorded['python']}" in environment_yml
    assert f"torch=={recorded['torch']}" in environment_yml
    for distribution, version in recorded.items():
        if distribution in {"python", "torch"}:
            continue
        assert f"{distribution}=={version}" in requirements
        assert f"{distribution}=={version}" in environment_yml


@pytest.mark.parametrize(
    "path",
    sorted(glob.glob(os.path.join(REPO, "demo", "fixtures", "*_expected.json"))),
    ids=lambda p: os.path.basename(p),
)
def test_reference_fixture_matches_its_checksum_and_is_self_consistent(path):
    """Each fixture is a published reference target; a corrupted one would send
    a reproducer chasing a phantom mismatch."""
    import hashlib
    import json

    manifest = json.load(open(os.path.join(REPO, "centroids", "MANIFEST.json")))
    rel = os.path.relpath(path, REPO)
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == manifest["files"][rel]["sha256"]

    d = json.load(open(path))
    suf, summ = d["sufficiency"], d["_summary"]
    assert set(suf) == set(PAPER_TASKS_EXPECTED), f"{rel}: unexpected task set"
    mt = sum(v["text_centroid_cost"] for v in suf.values()) / len(suf)
    mv = sum(v["vis_centroid_cost"] for v in suf.values()) / len(suf)
    assert mt == pytest.approx(summ["mean_text_cost"], abs=5e-4)
    assert mv == pytest.approx(summ["mean_vis_cost"], abs=5e-4)


PAPER_TASKS_EXPECTED = [
    "Forensic_Detection", "Visual_Similarity", "Art_Style",
    "Counting", "Relative_Depth", "Spatial_Relation",
]


def test_every_shipped_bank_has_a_reference_fixture():
    for b in BANKS:
        name = os.path.basename(b)[:-4]
        fx = os.path.join(REPO, "demo", "fixtures", f"{name}_expected.json")
        assert os.path.exists(fx), f"no published reference for bank '{name}'"


def test_published_asymmetry_is_text_dominant_for_every_model():
    """The paper's central claim, checked against the shipped reference data."""
    import json

    for b in BANKS:
        name = os.path.basename(b)[:-4]
        d = json.load(open(os.path.join(REPO, "demo", "fixtures", f"{name}_expected.json")))
        s = d["_summary"]
        assert s["mean_text_cost"] > s["mean_vis_cost"], f"{name} is not text-dominant"
        assert s["asymmetry_ratio"] > 1.0, f"{name} asymmetry {s['asymmetry_ratio']} <= 1"


# ── the shipped fixtures must reproduce the paper's headline table ──
#
# Table 1 (tab:asymmetry) of the paper, transcribed. A reader can check the
# release against the printed paper without needing our working repo.

PAPER_TABLE_1 = {
    # bank:        (mean visual cost %, mean text cost %, asymmetry)
    "qwen":        (1.4, 27.2, 19.2),
    "qwen_3b":     (9.1, 13.7, 1.5),
    "qwen3":       (16.1, 31.5, 2.0),
    "qwen3_4b":    (16.2, 34.7, 2.1),
    "internvl":    (-0.9, 29.4, 34.4),
    "llava_ov":    (5.2, 25.8, 4.9),
    "idefics3":    (-1.5, 18.9, 12.7),
}


@pytest.mark.parametrize("bank", sorted(PAPER_TABLE_1))
def test_fixture_reproduces_the_papers_headline_table(bank):
    import json

    pv, pt, pa = PAPER_TABLE_1[bank]
    s = json.load(
        open(os.path.join(REPO, "demo", "fixtures", f"{bank}_expected.json"))
    )["_summary"]
    assert s["mean_vis_cost"] * 100 == pytest.approx(pv, abs=0.06)
    assert s["mean_text_cost"] * 100 == pytest.approx(pt, abs=0.06)
    assert s["asymmetry_ratio"] == pytest.approx(pa, abs=0.06)


def test_the_papers_four_times_headline_holds_across_the_seven_models():
    """The abstract's '4x' is the seven-model mean, not a per-model claim."""
    import json

    ts, vs = [], []
    for bank in PAPER_TABLE_1:
        s = json.load(
            open(os.path.join(REPO, "demo", "fixtures", f"{bank}_expected.json"))
        )["_summary"]
        ts.append(s["mean_text_cost"])
        vs.append(s["mean_vis_cost"])
    mt, mv = sum(ts) / len(ts), sum(vs) / len(vs)
    assert mt / mv == pytest.approx(4.0, abs=0.3), f"seven-model ratio is {mt / mv:.2f}x"
